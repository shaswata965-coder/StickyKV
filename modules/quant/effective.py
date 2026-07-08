"""Effective K/V materialization — the two-tier read path (design.md §5, §8).

Phase 1 (materialize-then-interleave): dequantize the Q store, apply RoPE at
each Q window's **current** contiguous ``position_range``, and interleave with
the fp store **chronologically by window id** into one effective tensor for
the standard attention path. Attention itself is order-free (RoPE bakes each
key's logical position into its values), but the window scorer chunks the
physical key axis, so physical order must equal chronological window order.

The transient fp16 copy of the Q tier is per-layer and freed as soon as the
caller drops the returned tensors (§8) — nothing here caches across steps.

Also hosts the shared RoPE strip/apply helper used by the tier-crossing moves
(demotion un-rotates once; promotion rotates the dequantized pre-RoPE codes at
their interleaved target positions).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from .ledger import QuantLedger
from .quantizer import dequantize_key_windows, dequantize_value_windows
from .store import QuantizedStore


def _apply_rotary_pos_emb():
    """Resolve HF's rotary helper lazily (keeps this module import-light)."""
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb


def apply_rope_to_keys(
    rope_module: torch.nn.Module,
    keys: Tensor,
    position_ids: Tensor,
    inverse: bool = False,
) -> Tensor:
    """Rotate (or, with ``inverse=True``, un-rotate) *keys* at *position_ids*.

    Uses the model's own rope module + ``apply_rotary_pos_emb`` so NTK / YaRN
    scaling is preserved — the same math as ``CacheState.rerotate_keys``, split
    into its two halves so tier moves can strip once (demotion) or apply once
    (promotion / Q-tier read) without a paired pass.

    Parameters
    ----------
    keys : Tensor
        ``[B, H, T, D]``.
    position_ids : Tensor
        ``[B, T]`` or ``[T]`` (broadcast).
    inverse : bool
        ``True`` strips an existing rotation (cos(−θ)=cos θ, sin(−θ)=−sin θ).
    """
    rotary = _apply_rotary_pos_emb()
    B = keys.shape[0]
    pos = position_ids
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(B, -1)
    cos, sin = rope_module(keys, pos)
    if inverse:
        sin = -sin
    _, rotated = rotary(keys, keys, cos, sin)
    # HF rope modules return cos/sin in the input dtype, but a module that
    # emits higher precision would silently promote the keys here — pin the
    # output to the store dtype (a no-op when dtypes already match).
    return rotated.to(keys.dtype)


def materialize_effective_kv(
    state: Any,
    store: QuantizedStore,
    ledger: QuantLedger,
    rope_module: torch.nn.Module,
    window_size: int,
    keys_only: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Build the effective (fp ‖ dequantized-Q) K/V in chronological order.

    Steps (design §8): for the active Q windows — ledger lookup → dequantize
    the int4 codes against their pinned fp16 grids → apply RoPE at each
    window's current ``position_range`` → interleave with the fp store by
    ``original_window_id``. The interleave is a positional scatter: after
    every eviction the fp store's (gappy) ``position_ids`` and the Q windows'
    position ranges jointly tile ``arange(T_total)``, so writing each token at
    its position yields exactly the chronological merged order.

    Parameters
    ----------
    state : CacheState
        The layer's fp store (``key_states`` / ``value_states`` rotated and
        ready, ``position_ids`` = interleaved fp slots).
    store, ledger : QuantizedStore, QuantLedger
        The layer's Q tier. Dormant entries are excluded (design §6).
    rope_module : nn.Module
        The model's rotary embedding module.
    window_size : int
        Tokens per Q window.
    keys_only : bool
        Skip the value dequant/interleave (the flash score hook needs only K);
        the second element of the return is ``None``.

    Returns
    -------
    (k_eff, v_eff)
        Freshly-built tensors ``[B, H_kv, T_total, D]`` — **not** aliases of
        the stored fp cache (except on the fast path with zero active Q
        windows, where the live fp tensors are returned unchanged).
    """
    owids, slots, starts = ledger.active_view()
    n_q = owids.numel()
    if n_q == 0:
        return state.key_states, (None if keys_only else state.value_states)

    k_fp = state.key_states
    B, H_kv, T_fp, D = k_fp.shape
    S = window_size
    device = k_fp.device
    T_total = T_fp + n_q * S

    slots = slots.to(device)
    # Each Q window's contiguous position range, flattened to one token axis
    # in window-major (chronological) order.
    q_pos = (
        starts.to(device).unsqueeze(-1)
        + torch.arange(S, device=device, dtype=torch.long)
    ).view(-1)                                   # [n_q * S]
    fp_pos = state.position_ids[0].to(device)     # [T_fp] (B = 1 in v1)

    # Dequantize Q keys (stored pre-RoPE, channel-major) and stamp RoPE at the
    # current positions in one vectorized pass over all active windows.
    kc, ks, kz = store.gather_keys(slots)
    k_q = dequantize_key_windows(kc, ks, kz, out_dtype=k_fp.dtype)  # [n_q,H,S,D]
    k_q = k_q.permute(1, 0, 2, 3).reshape(1, H_kv, n_q * S, D)
    k_q = apply_rope_to_keys(rope_module, k_q, q_pos.unsqueeze(0))

    k_eff = k_fp.new_empty(B, H_kv, T_total, D)
    k_eff.index_copy_(2, fp_pos, k_fp)
    k_eff.index_copy_(2, q_pos, k_q)

    if keys_only:
        return k_eff, None

    v_fp = state.value_states
    vc, vs, vz = store.gather_values(slots)
    # Values interleave with the VALUE store — use its dtype (keys and values
    # can legitimately diverge if a non-dtype-faithful rope promoted the keys).
    v_q = dequantize_value_windows(vc, vs, vz, out_dtype=v_fp.dtype)  # [n_q,H,S,D]
    v_q = v_q.permute(1, 0, 2, 3).reshape(1, H_kv, n_q * S, D)
    v_eff = v_fp.new_empty(B, H_kv, T_total, D)
    v_eff.index_copy_(2, fp_pos, v_fp)
    v_eff.index_copy_(2, q_pos, v_q)
    return k_eff, v_eff
