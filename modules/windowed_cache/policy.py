"""EvictionPolicy — pure index / state-machine for windowed cache eviction.

Tracks region boundaries (sink, evictable, local) and the generation step
counter.  **Does not touch tensors** except via the ``window_scores`` input
to ``compute_retain_window_indices``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ResolvedConfig


def pool_over_heads(scores: Tensor, p: float) -> Tensor:
    """Reduce per-head window scores ``[B, H, W]`` to ``[B, W]`` via an Lp power-mean.

    ``p == 1`` is a plain mean over heads (``scores.mean(dim=1)``) — byte-identical
    to the historical behaviour. ``p > 1`` up-weights the heads that express a
    sharp preference for a window (retrieval heads) over heads that spread their
    attention diffusely, so a single head's evidence is not averaged away by many
    indifferent heads; ``p -> inf`` approaches a max over heads.

    The power-mean is normalised by the per-window head-max before the ``pow`` so
    that large ``p`` at full-context score magnitudes cannot overflow fp32. This
    is an exact identity: ``(mean_h s^p)^(1/p) = m · (mean_h (s/m)^p)^(1/p)`` with
    ``m = max_h s`` (``s >= 0`` always, since scores are rooted attention
    power-sums).
    """
    if p == 1.0:
        return scores.mean(dim=1)
    s = scores.to(torch.float32)
    m = s.amax(dim=1, keepdim=True).clamp_min(1e-12)   # [B, 1, W]
    pooled = (s / m).pow(p).mean(dim=1).pow(1.0 / p) * m.squeeze(1)  # [B, W]
    return pooled.to(scores.dtype)


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
        self.total_tokens: int = 0
        # Head-pooling knobs (default: plain mean over heads → byte-identical).
        self.score_p_head: float = resolved.score_p_head
        self.head_group_pool: str = resolved.head_group_pool
        self.num_key_value_groups: int = resolved.num_key_value_groups

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

    def _reduce_heads(self, window_scores: Tensor) -> Tensor:
        """Reduce per-head window scores ``[B, H_q, W]`` to ``[B, W]``.

        Two-stage, GQA-aware:

        1. If ``head_group_pool != "none"`` and the query heads divide evenly
           into ``num_key_value_groups``, pool the query heads **within each GQA
           group** (``"max"`` keeps what the group's most-demanding member needs;
           ``"mean"`` averages them) → one score per KV head.
        2. Pool the resulting heads with an Lp power-mean of exponent
           ``score_p_head`` (see :func:`pool_over_heads`).

        Defaults (``head_group_pool="none"``, ``score_p_head=1.0``) reduce to
        ``window_scores.mean(dim=1)``.
        """
        H_q = window_scores.shape[1]
        g = self.num_key_value_groups
        if self.head_group_pool != "none" and g > 1 and H_q % g == 0:
            B, _, W = window_scores.shape
            H_kv = H_q // g
            grouped = window_scores.view(B, H_kv, g, W)
            if self.head_group_pool == "max":
                per_head = grouped.amax(dim=2)   # [B, H_kv, W]
            else:  # "mean"
                per_head = grouped.mean(dim=2)   # [B, H_kv, W]
        else:
            per_head = window_scores             # [B, H_q, W]
        return pool_over_heads(per_head, self.score_p_head)

    def compute_retain_window_indices(
        self, window_scores: Tensor
    ) -> Tensor:
        """Primary retain-decision method at **window** granularity.

        Algorithm (all single-call tensor ops):
        1. ``mean_scores = self._reduce_heads(window_scores)`` → ``[B, W_total]``
           (Lp power-mean over heads, optionally GQA-group-aware; defaults to a
           plain mean).
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

        # 1. Reduce the head axis → [B, W_total]. Optionally pool query heads
        #    within their GQA group first (the query heads sharing one KV head
        #    must keep the same tokens, so with "max" the group keeps what its
        #    most-demanding member needs), then apply an Lp power-mean across the
        #    resulting heads. With head_group_pool="none" and score_p_head=1.0
        #    this is exactly window_scores.mean(dim=1).
        mean_scores = self._reduce_heads(window_scores)  # [B, W_total]

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
    # Retain indices — token granularity
    # -----------------------------------------------------------------

    def expand_to_token_indices(
        self, retained_window_idx: Tensor
    ) -> Tensor:
        """Expand window indices to absolute token indices.

        Prepends sink prefix.  Trims trailing partial window via geometric
        cap computed without touching tensor data (Python int arithmetic).

        Parameters
        ----------
        retained_window_idx : Tensor
            Shape ``[B, W_retained]``.

        Returns
        -------
        Tensor
            Shape ``[B, T_retained]``.
        """
        B, W_retained = retained_window_idx.shape
        device = retained_window_idx.device

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
        valid_mask = all_idx < self.total_tokens  # [B, total]

        # For batched gather we need rectangular tensors — count valid per row
        # and truncate to the minimum across the batch.
        valid_counts = valid_mask.sum(dim=1)          # [B]
        min_valid = int(valid_counts.min().item())

        # Gather only valid indices: sort valid-first via the mask, take prefix
        # argsort of ~mask (False=0 sorts before True=1) gives valid-idx-first order
        order = torch.argsort(~valid_mask, dim=1, stable=True)  # valid first
        all_idx = torch.gather(all_idx, 1, order)[:, :min_valid]

        return all_idx
