"""Hand-rolled KIVI-style affine int4 quantizer (design.md §2).

Numerics are pinned by the design and must not drift:

- ``scale = (mx − mn) / 15``, ``zero = mn`` (float offset, not an integer
  zero-point), computed in fp32 over the quant group, then **stored fp16**.
- Codes are fit against the **fp16-rounded** grid (not the fp32
  intermediates), so the grid the codes were fit to is bit-identical to the
  grid used at every dequant.
- ``q = clamp(round_half_even((x − zero) / scale), 0, 15)`` — clamp **before**
  the uint cast (``torch.round`` is round-half-even).
- ``x̂ = q · scale + zero``.
- Degenerate group (``mx == mn``): ``scale = 1``, all codes 0, ``x̂ = mn``
  exactly.

Granularity (design §2):

- **Keys** — per-``(head, channel, window)``: channel-major codes
  ``[N, H_kv, D, S]`` packed 2 tokens/byte → ``[N, H_kv, D, S/2]``;
  scale/zero ``[N, H_kv, D]`` per window.
- **Values** — per-token: token-major codes ``[N, H_kv, S, D]`` packed
  2 channels/byte → ``[N, H_kv, S, D/2]``; scale/zero ``[N, H_kv, S]``.

Nibble packing packs the two codes that **share a scale** into one byte
(pack along the quantization-group axis); even-index code in the low 4 bits.
``window_size`` must be even (asserted) so there is no tail padding.

Everything here is pure-tensor, CPU-testable, and transformers-free.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core affine grid / codes / dequant
# ---------------------------------------------------------------------------


def compute_grid(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
    """Compute the pinned fp16 affine grid over the group axis *dim*.

    ``scale = (mx − mn)/15`` and ``zero = mn`` are computed in fp32 and
    **rounded to fp16** — the returned tensors are the stored, pinned grid.
    Degenerate groups (``mx == mn``) get ``scale = 1``.

    Returns
    -------
    (scale, zero) : Tuple[Tensor, Tensor]
        fp16, with *dim* kept (size 1) for broadcasting.
    """
    xf = x.to(torch.float32)
    mn = xf.amin(dim=dim, keepdim=True)
    mx = xf.amax(dim=dim, keepdim=True)
    scale = (mx - mn) / 15.0
    scale = torch.where(mx == mn, torch.ones_like(scale), scale)
    return scale.to(torch.float16), mn.to(torch.float16)


def quantize_with_grid(x: Tensor, scale: Tensor, zero: Tensor) -> Tensor:
    """Fit int4 codes against an existing (fp16, pinned) grid.

    ``q = clamp(round_half_even((x − zero)/scale), 0, 15)`` with the arithmetic
    in fp32 **using the fp16-stored grid values**, clamped before the uint8
    cast. This function has no degenerate-group special case — use
    :func:`quantize` for first-time quantization; this exists for the
    fp16-grid idempotence property (re-quantizing a dequantized tensor against
    the same grid reproduces the codes bit-identically).
    """
    q = torch.round(
        (x.to(torch.float32) - zero.to(torch.float32)) / scale.to(torch.float32)
    ).clamp_(0.0, 15.0)
    return q.to(torch.uint8)


def quantize(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
    """First-time quantization of *x* over group axis *dim*.

    Computes the fp16 pinned grid, then fits codes against that fp16 grid.
    Degenerate groups (``mx == mn``) produce all-zero codes so
    ``dequantize`` reconstructs ``mn`` exactly (design §2).

    Returns
    -------
    (codes, scale, zero)
        ``codes`` uint8 (unpacked, one 4-bit value per byte), ``scale`` /
        ``zero`` fp16 with *dim* kept for broadcasting.
    """
    xf = x.to(torch.float32)
    mn = xf.amin(dim=dim, keepdim=True)
    mx = xf.amax(dim=dim, keepdim=True)
    degenerate = mx == mn
    scale = torch.where(degenerate, torch.ones_like(mx), (mx - mn) / 15.0)
    scale16 = scale.to(torch.float16)
    zero16 = mn.to(torch.float16)
    codes = quantize_with_grid(x, scale16, zero16)
    # Degenerate groups are defined to be all-zero codes regardless of any
    # fp16 rounding of `zero` (guarantees x̂ = mn exactly at dequant).
    codes = torch.where(degenerate, torch.zeros_like(codes), codes)
    return codes, scale16, zero16


def dequantize(
    codes: Tensor, scale: Tensor, zero: Tensor, out_dtype: torch.dtype = torch.float16
) -> Tensor:
    """``x̂ = q·scale + zero`` in fp32 against the pinned fp16 grid, cast to
    *out_dtype*."""
    x = codes.to(torch.float32) * scale.to(torch.float32) + zero.to(torch.float32)
    return x.to(out_dtype)


# ---------------------------------------------------------------------------
# Nibble packing (2 codes sharing a scale per byte; even index = low nibble)
# ---------------------------------------------------------------------------


def pack_nibbles(codes: Tensor) -> Tensor:
    """Pack unpacked uint8 codes (values 0–15) along the **last** axis.

    The last axis is the quantization-group axis (both nibbles of every byte
    share one scale). Its size must be even — ``window_size`` even is a design
    requirement (head_dim always is).
    """
    n = codes.shape[-1]
    assert n % 2 == 0, (
        f"pack axis must be even (got {n}); window_size must be even (design §2)"
    )
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(packed: Tensor) -> Tensor:
    """Inverse of :func:`pack_nibbles` — restore the unpacked code layout."""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)


# ---------------------------------------------------------------------------
# Window-level K / V quantization (the shapes the cache uses)
# ---------------------------------------------------------------------------


def quantize_key_windows(k_pre: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize pre-RoPE key windows per ``(head, channel, window)``.

    Parameters
    ----------
    k_pre : Tensor
        ``[N, H_kv, S, D]`` — N windows of S (= ``window_size``, even) tokens,
        keys **pre-RoPE**.

    Returns
    -------
    (codes, scale, zero)
        ``codes`` ``[N, H_kv, D, S/2]`` uint8 (channel-major, 2 tokens/byte);
        ``scale`` / ``zero`` ``[N, H_kv, D]`` fp16.
    """
    x = k_pre.permute(0, 1, 3, 2)  # channel-major [N, H_kv, D, S]
    codes, scale, zero = quantize(x, dim=-1)
    return pack_nibbles(codes), scale.squeeze(-1), zero.squeeze(-1)


def dequantize_key_windows(
    codes: Tensor, scale: Tensor, zero: Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of :func:`quantize_key_windows` → pre-RoPE keys ``[N, H_kv, S, D]``."""
    unpacked = unpack_nibbles(codes)  # [N, H_kv, D, S]
    x = dequantize(unpacked, scale.unsqueeze(-1), zero.unsqueeze(-1), out_dtype)
    return x.permute(0, 1, 3, 2).contiguous()  # [N, H_kv, S, D]


def quantize_value_windows(v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantize value windows per token.

    Parameters
    ----------
    v : Tensor
        ``[N, H_kv, S, D]`` (``D`` even — head_dim always is).

    Returns
    -------
    (codes, scale, zero)
        ``codes`` ``[N, H_kv, S, D/2]`` uint8 (token-major, 2 channels/byte);
        ``scale`` / ``zero`` ``[N, H_kv, S]`` fp16.
    """
    codes, scale, zero = quantize(v, dim=-1)
    return pack_nibbles(codes), scale.squeeze(-1), zero.squeeze(-1)


def dequantize_value_windows(
    codes: Tensor, scale: Tensor, zero: Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Inverse of :func:`quantize_value_windows` → values ``[N, H_kv, S, D]``."""
    unpacked = unpack_nibbles(codes)  # [N, H_kv, S, D]
    return dequantize(unpacked, scale.unsqueeze(-1), zero.unsqueeze(-1), out_dtype)
