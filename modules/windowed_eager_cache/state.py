"""CacheState — tensor storage for a single attention layer's KV cache.

Layout follows original sequence order: ``[sink | evictable | local]``.
Region boundaries are tracked by :class:`EvictionPolicy`, not here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class CacheState:
    """Mutable tensor state for one layer's KV cache.

    Attributes
    ----------
    key_states : Tensor
        Shape ``[B, H_kv, T, D]``.
    value_states : Tensor
        Shape ``[B, H_kv, T, D]``.
    position_ids : Tensor
        Shape ``[B, T]``, int64.  Eviction only **compacts**: surviving keys
        keep the RoPE rotation they were given at their *original* absolute
        positions, and ``slice_and_keep`` gathers those original positions so
        the surviving tokens carry their true positions (just packed
        contiguously in memory).  RoPE is never stripped or re-applied, and the
        query keeps its natural (monotonic, absolute) position from HF, so the
        query<->key relative phase is preserved without any override.
    window_scores : Tensor
        Shape ``[B, H_q, W]``.  Running cumulative per-window scores.
    original_window_ids : Tensor
        Shape ``[B, W]``, int64.  Maps each surviving compact window index to
        its original sequence window index (0-based after sinks), **per row**.
        Stays identity before the first eviction; gathered per row alongside
        ``window_scores`` at every subsequent eviction so that compact
        top-K indices can be translated back to original positions for
        faithful Jaccard comparison.
    """

    __slots__ = ("key_states", "value_states", "position_ids",
                 "window_scores", "original_window_ids")

    def __init__(self) -> None:
        self.key_states: Optional[Tensor] = None
        self.value_states: Optional[Tensor] = None
        self.position_ids: Optional[Tensor] = None
        self.window_scores: Optional[Tensor] = None
        self.original_window_ids: Optional[Tensor] = None

    # -----------------------------------------------------------------
    # seq_length property
    # -----------------------------------------------------------------

    @property
    def seq_length(self) -> int:
        """Current sequence length (number of cached tokens)."""
        if self.key_states is None:
            return 0
        return self.key_states.shape[2]

    # -----------------------------------------------------------------
    # append
    # -----------------------------------------------------------------

    def append(
        self,
        key: Tensor,
        value: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> None:
        """Append new key/value states along the sequence dimension.

        Parameters
        ----------
        key : Tensor
            Shape ``[B, H_kv, N_new, D]``.
        value : Tensor
            Shape ``[B, H_kv, N_new, D]``.
        position_ids : Tensor, optional
            Shape ``[N_new]`` (shared across the batch, e.g. HF's
            ``cache_position``) or ``[B, N_new]`` (per row).  If ``None``,
            auto-increments from the current length.  Stored canonically as
            ``[B, N_new]``.
        """
        if self.key_states is None:
            # Take ownership: a contiguous clone prevents the cache from
            # aliasing caller-owned tensors that may be mutated later.
            self.key_states = key.contiguous().clone()
            self.value_states = value.contiguous().clone()
        else:
            self.key_states = torch.cat([self.key_states, key], dim=2)
            self.value_states = torch.cat([self.value_states, value], dim=2)

        n_new = key.shape[2]
        B = key.shape[0]
        device = key.device
        # position_ids is canonical as [B, N_new]. A 1-D input is shared across
        # the batch and broadcast; a [B, N_new] input is used as-is.
        if position_ids is not None:
            new_pos = position_ids
            if new_pos.dim() == 1:
                new_pos = new_pos.unsqueeze(0).expand(B, -1)
        else:
            start = 0 if self.position_ids is None else self.position_ids.shape[1]
            new_pos = (
                torch.arange(start, start + n_new, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )

        if self.position_ids is None:
            self.position_ids = new_pos.contiguous()
        else:
            self.position_ids = torch.cat(
                [self.position_ids, new_pos.to(self.position_ids.device)], dim=1
            )

    # -----------------------------------------------------------------
    # slice_and_keep
    # -----------------------------------------------------------------

    def slice_and_keep(self, retain_token_indices: Tensor) -> None:
        """Compact the cache by keeping only the tokens at *retain_token_indices*.

        Uses ``torch.gather`` with ``.expand()`` — never ``.repeat()``.

        Parameters
        ----------
        retain_token_indices : Tensor
            Shape ``[B, T_retained]``, int64 indices into the seq dimension.
        """
        B, T_retained = retain_token_indices.shape
        H_kv = self.key_states.shape[1]
        D = self.key_states.shape[3]

        # Expand indices for gather: [B, H_kv, T_retained, D]
        idx_k = (
            retain_token_indices
            .unsqueeze(1)   # [B, 1, T_retained]
            .unsqueeze(3)   # [B, 1, T_retained, 1]
            .expand(B, H_kv, T_retained, D)
        )

        self.key_states = torch.gather(self.key_states, dim=2, index=idx_k).contiguous()
        self.value_states = torch.gather(self.value_states, dim=2, index=idx_k).contiguous()

        # Gather position_ids to the surviving tokens' ORIGINAL positions.
        # Keys are NOT re-rotated: survivors retain the RoPE rotation baked in
        # at their original absolute positions, so position_ids must stay at
        # those original values (just compacted). position_ids is [B, T]; gather
        # each row independently because rows may evict different windows.
        if self.position_ids is not None:
            self.position_ids = torch.gather(
                self.position_ids, 1, retain_token_indices.to(self.position_ids.device)
            ).contiguous()
