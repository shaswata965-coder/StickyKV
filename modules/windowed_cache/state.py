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
        Shape ``[T]``, int64.  Rebased to ``arange(T)`` after eviction.
    window_scores : Tensor
        Shape ``[B, H_q, W]``.  Running cumulative per-window scores.
    """

    __slots__ = ("key_states", "value_states", "position_ids", "window_scores")

    def __init__(self) -> None:
        self.key_states: Optional[Tensor] = None
        self.value_states: Optional[Tensor] = None
        self.position_ids: Optional[Tensor] = None
        self.window_scores: Optional[Tensor] = None

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
            Shape ``[N_new]``.  If ``None``, auto-increments from current length.
        """
        if self.key_states is None:
            self.key_states = key
            self.value_states = value
        else:
            self.key_states = torch.cat([self.key_states, key], dim=2)
            self.value_states = torch.cat([self.value_states, value], dim=2)

        n_new = key.shape[2]
        device = key.device
        if position_ids is not None:
            new_pos = position_ids
        else:
            start = 0 if self.position_ids is None else self.position_ids.shape[0]
            new_pos = torch.arange(start, start + n_new, device=device, dtype=torch.long)

        if self.position_ids is None:
            self.position_ids = new_pos
        else:
            self.position_ids = torch.cat([self.position_ids, new_pos])

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

        # Rebase position_ids to contiguous [0..T_retained-1]
        self.position_ids = torch.arange(
            T_retained, device=self.key_states.device, dtype=torch.long
        )

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
            Shape ``[T_retained]``, the original positions before compaction.
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

        T_retained = self.key_states.shape[2]
        device = self.key_states.device

        # Old positions → cos/sin
        old_pos = old_position_ids.unsqueeze(0)  # [1, T]
        cos_old, sin_old = rope_module(self.value_states, old_pos)

        # Undo old rotation: cos(-θ)=cos(θ), sin(-θ)=-sin(θ)
        _, k_unrotated = apply_rotary_pos_emb(
            self.key_states, self.key_states, cos_old, -sin_old
        )

        # New contiguous positions
        new_pos = torch.arange(T_retained, device=device, dtype=torch.long).unsqueeze(0)
        cos_new, sin_new = rope_module(self.value_states, new_pos)

        # Apply new rotation
        _, k_rerotated = apply_rotary_pos_emb(
            k_unrotated, k_unrotated, cos_new, sin_new
        )

        self.key_states = k_rerotated
