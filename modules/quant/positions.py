"""Interleaved position map across the fp16 and int4 tiers (design.md §5).

Every eviction compacts BOTH tiers jointly: all surviving windows are sorted
by ``original_window_id`` and assigned one contiguous position map
``arange(T_total)`` (after the sink prefix). Fp windows take their slots in
that map — skipping over the positions occupied by interleaved Q windows —
and each Q window gets a contiguous ``position_range`` for its ledger entry.

Per-row by construction (rows may evict different windows), vectorized over
the batch axis. Shared by both cache backends.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def build_interleaved_position_map(
    fp_window_ids: Tensor,
    q_window_ids: Tensor,
    window_size: int,
    num_sink: int,
    fp_tail_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Assign contiguous interleaved positions to all surviving windows.

    Parameters
    ----------
    fp_window_ids : Tensor
        ``[B, W_fp]`` int64 — surviving fp-tier ``original_window_id``s,
        ascending per row (chronological).
    q_window_ids : Tensor
        ``[B, W_q]`` int64 — surviving (active) Q-tier ids, ascending per row.
    window_size : int
        Tokens per window (Q windows are always full).
    num_sink : int
        Sink prefix length — sink tokens occupy positions ``[0, num_sink)``
        ahead of the window map (always fp, never scored).
    fp_tail_len : int, optional
        Length of the **last** fp window when it is partial (< window_size).
        A partial window can only be the globally newest window (the trailing
        local window), which is always fp — asserted.

    Returns
    -------
    (fp_positions, q_position_starts) : Tuple[Tensor, Tensor]
        ``fp_positions`` ``[B, T_fp]`` — target position of every fp-store
        token (sink prefix included), gappy where Q windows interleave; feed
        to ``rerotate_keys(..., new_position_ids=...)``.
        ``q_position_starts`` ``[B, W_q]`` — first position of each Q window's
        contiguous range, ordered like *q_window_ids*; feed to the ledger.
    """
    B, W_fp = fp_window_ids.shape
    W_q = q_window_ids.shape[1]
    S = window_size
    device = fp_window_ids.device
    tail = S if fp_tail_len is None else fp_tail_len
    assert 0 < tail <= S, f"fp_tail_len must be in (0, window_size], got {tail}"
    assert W_fp > 0, "fp tier cannot be empty (local windows are always fp)"

    # Merge both tiers and sort by original_window_id (stable not needed —
    # a window id lives in exactly one tier).
    merged_ids = torch.cat([fp_window_ids, q_window_ids], dim=1)  # [B, W]
    is_fp = torch.cat(
        [
            torch.ones(B, W_fp, dtype=torch.bool, device=device),
            torch.zeros(B, W_q, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    order = torch.argsort(merged_ids, dim=1)
    is_fp_sorted = torch.gather(is_fp, 1, order)  # [B, W]

    # Window sizes along the sorted (chronological) axis: all full except a
    # partial trailing fp window, which is by definition the newest id → the
    # last sorted slot.
    W = W_fp + W_q
    sizes = torch.full((B, W), S, dtype=torch.long, device=device)
    if tail < S:
        assert bool(is_fp_sorted[:, -1].all()), (
            "a partial trailing window must be the newest window and fp-tier"
        )
        sizes[:, -1] = tail

    # Exclusive cumsum → each sorted window's start position (after the sink).
    starts_sorted = num_sink + torch.cumsum(sizes, dim=1) - sizes  # [B, W]

    # Q windows: sorted order restricted to the Q tier is ascending-id, which
    # is exactly the *q_window_ids* order.
    q_position_starts = starts_sorted[~is_fp_sorted].view(B, W_q)

    # Fp windows: expand each start to its token positions and trim the tail.
    fp_starts = starts_sorted[is_fp_sorted].view(B, W_fp)  # [B, W_fp]
    offsets = torch.arange(S, device=device, dtype=torch.long)
    fp_tok = (fp_starts.unsqueeze(-1) + offsets).reshape(B, W_fp * S)
    t_fp_post_sink = (W_fp - 1) * S + tail
    fp_tok = fp_tok[:, :t_fp_post_sink]

    sink_pos = (
        torch.arange(num_sink, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(B, -1)
    )
    fp_positions = torch.cat([sink_pos, fp_tok], dim=1)  # [B, T_fp]
    return fp_positions, q_position_starts
