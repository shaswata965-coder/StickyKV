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
        self._generation_step: int = 0

    # -----------------------------------------------------------------
    # State bookkeeping
    # -----------------------------------------------------------------

    def initialize_after_prefill(self, prefill_len: int) -> None:
        """Set state after the initial prefill pass."""
        self.total_tokens = prefill_len

    def extend_total_after_append(self, n_new: int) -> None:
        """Update total token count after appending *n_new* tokens."""
        self.total_tokens += n_new

    def slide_local_window(self) -> None:
        """Advance the local window boundary after a generation step."""
        # Region boundaries are recalculated on-the-fly from total_tokens
        pass

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

        # Trim: geometric cap via Python int arithmetic (no tensor data read)
        max_tokens = self.total_tokens
        retained_len = min(all_idx.shape[1], max_tokens)
        all_idx = all_idx[:, :retained_len]

        return all_idx

    # -----------------------------------------------------------------
    # Convenience: combined
    # -----------------------------------------------------------------

    def compute_retain_token_indices(
        self, window_scores: Tensor
    ) -> Tensor:
        """Convenience: ``expand_to_token_indices(compute_retain_window_indices(...))``.

        The cache **must** use the two-step form so it can also gather
        ``state.window_scores`` by ``retained_window_idx``.
        """
        retained_win = self.compute_retain_window_indices(window_scores)
        return self.expand_to_token_indices(retained_win)
