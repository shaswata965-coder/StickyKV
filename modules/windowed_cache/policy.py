"""EvictionPolicy — pure index / state-machine for windowed cache eviction.

Tracks region boundaries (sink, evictable, local) and the generation step
counter.  **Does not touch tensors** except via the ``window_scores`` input
to ``compute_retain_window_indices``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ResolvedConfig


class EvictionPolicy:
    """Stateful eviction controller.

    Parameters
    ----------
    resolved : ResolvedConfig
        Resolved configuration with concrete integer counts.
    """

    def __init__(self, resolved: ResolvedConfig) -> None:
        self.window_size: int = resolved.window_size
        self.num_sink_tokens: int = resolved.num_sink_tokens
        self.local_tokens: int = resolved.local_tokens
        self.local_windows: int = resolved.local_tokens // resolved.window_size
        self.top_k_windows: int = resolved.top_k_windows
        self.top_q_windows: int = resolved.top_q_windows
        self.total_tokens: int = 0

    # -----------------------------------------------------------------
    # State bookkeeping
    # -----------------------------------------------------------------

    def initialize_after_prefill(self, prefill_len: int) -> None:
        """Set state after the initial prefill pass."""
        self.total_tokens = prefill_len

    def extend_total_after_append(self, n_new: int) -> None:
        """Update total token count after appending *n_new* tokens."""
        self.total_tokens += n_new

    def set_total_after_compaction(self, new_total: int) -> None:
        """Update state after eviction compaction."""
        self.total_tokens = new_total

    # -----------------------------------------------------------------
    # Eviction trigger
    # -----------------------------------------------------------------

    def should_evict(self, step: int) -> bool:
        """Return ``True`` if eviction should run at *step*."""
        return step > 0 and step % self.window_size == 0

    # -----------------------------------------------------------------
    # Region helpers (computed from current state)
    # -----------------------------------------------------------------

    @property
    def post_sink_tokens(self) -> int:
        return max(self.total_tokens - self.num_sink_tokens, 0)

    @property
    def num_total_windows(self) -> int:
        ps = self.post_sink_tokens
        return (ps + self.window_size - 1) // self.window_size if ps > 0 else 0

    @property
    def num_evictable_windows(self) -> int:
        return max(self.num_total_windows - self.local_windows, 0)

    # -----------------------------------------------------------------
    # Retain indices — window granularity
    # -----------------------------------------------------------------

    def compute_retain_window_indices(
        self, window_scores: Tensor
    ) -> Tensor:
        """Primary retain-decision method at **window** granularity.

        Algorithm (all single-call tensor ops):
        1. ``mean_scores = window_scores.mean(dim=1)`` → ``[B, W_total]``.
        2. Slice to evictable window range.
        3. ``torch.topk`` on the slice.
        4. Sort indices chronologically (never by score).
        5. ``cat([sorted_topk_idx, local_window_idx], dim=-1)``

        Parameters
        ----------
        window_scores : Tensor
            Shape ``[B, H_q, W_total]``.

        Returns
        -------
        Tensor
            Shape ``[B, W_retained]``, window indices to keep.
        """
        B = window_scores.shape[0]
        W_total = window_scores.shape[2]
        device = window_scores.device

        # Number of local and evictable windows
        local_w = min(self.local_windows, W_total)
        evictable_w = W_total - local_w

        # 1. Mean across heads
        mean_scores = window_scores.mean(dim=1)  # [B, W_total]

        # 2. Slice to evictable window range [0, evictable_w)
        evictable_scores = mean_scores[:, :evictable_w]  # [B, evictable_w]

        # Edge case: if num_evictable ≤ top_k, retain all evictable
        k = min(self.top_k_windows, evictable_w)

        if k == 0 or evictable_w == 0:
            # No evictable windows to select — just keep local
            local_idx = torch.arange(
                W_total - local_w, W_total, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            return local_idx

        if k >= evictable_w:
            # Keep all evictable + all local
            all_idx = torch.arange(
                W_total, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            return all_idx

        # 3. Top-K on evictable slice
        _, topk_idx = torch.topk(evictable_scores, k, dim=-1)  # [B, k]

        # 4. Sort indices chronologically
        topk_sorted, _ = torch.sort(topk_idx, dim=-1)

        # 5. Concatenate with local window indices
        local_idx = torch.arange(
            W_total - local_w, W_total, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        retained = torch.cat([topk_sorted, local_idx], dim=-1)  # [B, k + local_w]
        return retained

    # -----------------------------------------------------------------
    # Retain indices — two-tier (fp16 + int4) window granularity
    # -----------------------------------------------------------------

    def compute_tier_assignments(
        self, window_scores: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Two-tier retain decision on the **merged** window axis (design §5
        steps 1–2).

        Ranks the evictable band once and splits the ranking: the top
        ``top_k_windows`` (plus the protected local windows) go to the fp
        tier; the **next** ``top_q_windows`` go to the int4 Q tier; the rest
        are dropped. Pure score-ranking arithmetic — no dequantization is
        needed for the decision (design §3).

        Parameters
        ----------
        window_scores : Tensor
            Shape ``[B, H_q, W_total]`` on the merged window axis.

        Returns
        -------
        (fp_retained, q_retained) : Tuple[Tensor, Tensor]
            ``[B, W_fp]`` and ``[B, W_q]`` merged-axis window indices, each
            sorted chronologically per row.  ``fp_retained`` includes the
            local windows.
        """
        B = window_scores.shape[0]
        W_total = window_scores.shape[2]
        device = window_scores.device

        local_w = min(self.local_windows, W_total)
        evictable_w = W_total - local_w

        local_idx = torch.arange(
            W_total - local_w, W_total, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        k_fp = min(self.top_k_windows, evictable_w)
        k_q = min(self.top_q_windows, evictable_w - k_fp)
        empty = torch.empty(B, 0, device=device, dtype=torch.long)

        if evictable_w == 0 or (k_fp == 0 and k_q == 0):
            return local_idx, empty

        mean_scores = window_scores.mean(dim=1)          # [B, W_total]
        evictable_scores = mean_scores[:, :evictable_w]  # [B, evictable_w]

        # One ranking, split at the tier boundary: topk is score-descending,
        # so the first k_fp indices are the fp band, the next k_q the Q band.
        _, top_idx = torch.topk(evictable_scores, k_fp + k_q, dim=-1)
        fp_sorted, _ = torch.sort(top_idx[:, :k_fp], dim=-1)
        q_sorted, _ = torch.sort(top_idx[:, k_fp:], dim=-1)

        fp_retained = torch.cat([fp_sorted, local_idx], dim=-1)
        return fp_retained, q_sorted

    # -----------------------------------------------------------------
    # Retain indices — token granularity
    # -----------------------------------------------------------------

    def expand_to_token_indices(
        self, retained_window_idx: Tensor, total_tokens: Optional[int] = None
    ) -> Tensor:
        """Expand window indices to absolute token indices.

        Prepends sink prefix.  Trims trailing partial window via geometric
        cap computed without touching tensor data (Python int arithmetic).

        Tier-aware use (design §5): with a live Q tier, merged-axis indices no
        longer map to physical offsets by arithmetic alone.  The caller
        translates the retained **fp partition** to fp-store window *ranks*
        (cumsum over the fp-tier mask) and passes those ranks here together
        with the explicit fp-store ``total_tokens`` cap; Q windows resolve
        through the ledger and get **no** token gather.

        Parameters
        ----------
        retained_window_idx : Tensor
            Shape ``[B, W_retained]`` — physical fp-store window indices
            (identical to merged indices on the single-tier path).
        total_tokens : int, optional
            Token count bounding the store being indexed.  Defaults to
            ``self.total_tokens`` (the single-tier path, byte-identical).

        Returns
        -------
        Tensor
            Shape ``[B, T_retained]``.
        """
        B, W_retained = retained_window_idx.shape
        device = retained_window_idx.device
        if total_tokens is None:
            total_tokens = self.total_tokens

        # Sink prefix [0, 1, ..., num_sink-1]
        sink_idx = torch.arange(
            self.num_sink_tokens, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)  # [B, num_sink]

        # Expand windows to tokens:
        # For window w: tokens = num_sink + w * window_size + offset
        offsets = torch.arange(
            self.window_size, device=device, dtype=torch.long
        )  # [window_size]

        # [B, W_retained, 1] * window_size + [window_size] → [B, W_retained, window_size]
        token_idx = (
            self.num_sink_tokens
            + retained_window_idx.unsqueeze(-1) * self.window_size
            + offsets
        )
        token_idx = token_idx.reshape(B, -1)  # [B, W_retained * window_size]

        # Concatenate sink + window tokens
        all_idx = torch.cat([sink_idx, token_idx], dim=-1)  # [B, total]

        # Mask out indices that exceed the actual sequence length
        # (partial last window produces OOB token positions).
        valid_mask = all_idx < total_tokens  # [B, total]

        # For batched gather we need rectangular tensors — count valid per row
        # and truncate to the minimum across the batch.
        valid_counts = valid_mask.sum(dim=1)          # [B]
        min_valid = int(valid_counts.min().item())

        # Gather only valid indices: sort valid-first via the mask, take prefix
        # argsort of ~mask (False=0 sorts before True=1) gives valid-idx-first order
        order = torch.argsort(~valid_mask, dim=1, stable=True)  # valid first
        all_idx = torch.gather(all_idx, 1, order)[:, :min_valid]

        return all_idx
