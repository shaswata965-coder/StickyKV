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
        Shape ``[B, T]``, int64.  Every eviction compacts then **re-rotates**:
        ``slice_and_keep`` first gathers the surviving tokens' original
        positions (so ``rerotate_keys`` can strip RoPE with the correct angles),
        then ``rerotate_keys`` rebases them to contiguous ``arange(T_retained)``
        and re-applies RoPE at those positions (KVPress methodology).  The query
        position is overridden to the compacted cache length each step
        (``install_position_override_hook``), keeping relative phase exact.
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
        # This is the intermediate state: the caller has already snapshotted
        # these originals, and rerotate_keys (called right after) strips RoPE at
        # these angles and then rebases position_ids to contiguous
        # arange(T_retained). position_ids is [B, T]; gather each row
        # independently because rows may evict different windows.
        if self.position_ids is not None:
            self.position_ids = torch.gather(
                self.position_ids, 1, retain_token_indices.to(self.position_ids.device)
            ).contiguous()

    # -----------------------------------------------------------------
    # rerotate_keys
    # -----------------------------------------------------------------

    def rerotate_keys(
        self,
        rope_module: torch.nn.Module,
        old_position_ids: Tensor,
    ) -> None:
        """Strip old RoPE rotation and re-apply with contiguous positions.

        Uses the model's own ``apply_rotary_pos_emb`` to preserve NTK / YaRN
        scaling.  Values are **not** rotated (RoPE applies to keys only in
        LLaMA / Qwen).

        Parameters
        ----------
        rope_module : nn.Module
            The model's rotary embedding module (e.g. ``model.model.rotary_emb``).
        old_position_ids : Tensor
            Shape ``[B, T_retained]`` (per row), or ``[T_retained]`` (shared
            across the batch, broadcast).  The original positions before
            compaction.
        """
        # Lazy import to avoid circular deps at module level
        try:
            from transformers.models.llama.modeling_llama import (
                apply_rotary_pos_emb,
            )
        except ImportError:
            from transformers.models.qwen2.modeling_qwen2 import (
                apply_rotary_pos_emb,
            )

        B = self.key_states.shape[0]
        T_retained = self.key_states.shape[2]
        device = self.key_states.device

        # Old positions → cos/sin.  Accept a shared 1-D vector or per-row [B, T].
        old_pos = old_position_ids
        if old_pos.dim() == 1:
            old_pos = old_pos.unsqueeze(0).expand(B, -1)
        cos_old, sin_old = rope_module(self.value_states, old_pos)

        # Undo old rotation: cos(-θ)=cos(θ), sin(-θ)=-sin(θ)
        _, k_unrotated = apply_rotary_pos_emb(
            self.key_states, self.key_states, cos_old, -sin_old
        )

        # New contiguous positions (same for every row)
        new_pos = (
            torch.arange(T_retained, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )
        cos_new, sin_new = rope_module(self.value_states, new_pos)

        # Apply new rotation
        _, k_rerotated = apply_rotary_pos_emb(
            k_unrotated, k_unrotated, cos_new, sin_new
        )

        self.key_states = k_rerotated
        # Keys now live at contiguous positions [0..T_retained-1]; keep the
        # bookkeeping in sync (slice_and_keep left the *original* positions) so
        # a subsequent eviction snapshots correct "old" positions.
        self.position_ids = (
            torch.arange(T_retained, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
