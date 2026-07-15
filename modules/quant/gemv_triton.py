"""Triton fused GEMV decode kernel — dequant-inside-attention (design.md §11, Phase 2).

The Phase-2 endpoint of the roadmap: a GPU kernel that loads the int4 Q-tier
codes **tile-by-tile**, dequantizes to fp16 **in registers**, applies RoPE from
cos/sin on the already-loaded data, and MACs against the query — so the fp16
blow-up is never written to HBM; **only int4 codes are read from HBM**. Tile
boundary = window boundary = scale-group boundary (one tile reads one window's
codes and its one pinned ``(scale, zero)``), exactly as §11 specifies.

Layout of this module
---------------------
- :func:`triton_available` / dispatch guard.
- :func:`_marshal_q_windows` — host-side gather of the active ledger entries into
  contiguous int4 code / grid / cos-sin buffers (no fp blow-up; a stand-in for
  the design's compacting dense store). **CPU-testable.**
- :func:`gemv_decode_reference_from_marshalled` — a pure-torch **mirror** of the
  kernel's exact arithmetic (nibble parity, per-channel key grid, per-token value
  grid, contiguous-half RoPE, even/odd value unpack, GQA head mapping, online
  softmax). It runs on CPU and is asserted equal to the streaming oracle
  (:func:`modules.quant.gemv.tiled_gemv_attention`), so the kernel's *algorithm*
  is validated without a GPU; the Triton code below is a line-for-line
  transcription of this mirror. **CPU-testable.**
- ``_gemv_decode_kernel`` + :func:`gemv_decode_triton` — the Triton kernel and its
  launcher. Requires CUDA + triton; **not yet run/validated** (this dev box is
  CPU-only). Validate on a GPU box with ``pytest -m gpu tests/test_gemv_triton.py``.

Numerics match the reference/oracle: fp32 accumulation, RoPE via the model's own
cos/sin, pinned fp16 grid. Scope: decode (single query token), B = 1, GQA.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from .effective import _rope_cos_sin  # shared RoPE cos/sin (model's own scaling)


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


def triton_available() -> bool:
    """True iff a Triton GPU path can run (triton importable **and** CUDA present)."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Host-side marshalling of the active Q tier (int4 only — no fp blow-up)
# ---------------------------------------------------------------------------


def _marshal_q_windows(
    store, rope_module: torch.nn.Module, ref: Tensor
) -> Optional[Dict[str, Tensor]]:
    """Gather active Q windows into contiguous int4/grid/cos-sin buffers.

    Only int4 codes + fp16 grids + cos/sin (position data) are marshalled — the
    fp16 key/value blow-up never happens here; that is the kernel's job, tile by
    tile. In a production layout the store would already be one contiguous packed
    buffer (design's Phase-2 dense store); stacking the ledger entries stands in
    for that.

    ``ref`` supplies device/dtype for the RoPE cos/sin. Returns ``None`` when the
    Q tier is empty.

    Returned tensors (``N`` = active windows, ``H`` = kv heads, ``ws`` = window):

    - ``key_codes``  : ``[N, H, D, ws//2]`` uint8 — channel-major, 2 tokens/byte.
    - ``key_scale`` / ``key_zero`` : ``[N, H, D]`` fp16 — per (head, channel).
    - ``val_codes``  : ``[N, H, ws, D//2]`` uint8 — token-major, 2 channels/byte.
    - ``val_scale`` / ``val_zero`` : ``[N, H, ws]`` fp16 — per (head, token).
    - ``cos`` / ``sin`` : ``[N, ws, D]`` — RoPE for each window's frozen positions.
    """
    entries = store.ledger.active_entries()
    if not entries:
        return None

    key_codes = torch.stack([e.key_codes for e in entries], dim=0)
    key_scale = torch.stack([e.key_scale for e in entries], dim=0)
    key_zero = torch.stack([e.key_zero for e in entries], dim=0)
    val_codes = torch.stack([e.val_codes for e in entries], dim=0)
    val_scale = torch.stack([e.val_scale for e in entries], dim=0)
    val_zero = torch.stack([e.val_zero for e in entries], dim=0)

    # cos/sin for every window's frozen positions, in one shot: [N*ws] -> [N,ws,D].
    ws = store.window_size
    positions = torch.stack([e.position_range.to(torch.long) for e in entries], dim=0)  # [N, ws]
    flat_pos = positions.reshape(-1)                                   # [N*ws]
    cos, sin = _rope_cos_sin(rope_module, ref, flat_pos)               # [1, N*ws, D]
    D = cos.shape[-1]
    cos = cos.reshape(len(entries), ws, D)
    sin = sin.reshape(len(entries), ws, D)

    return dict(
        key_codes=key_codes, key_scale=key_scale, key_zero=key_zero,
        val_codes=val_codes, val_scale=val_scale, val_zero=val_zero,
        cos=cos, sin=sin,
    )


# ---------------------------------------------------------------------------
# Pure-torch mirror of the kernel (CPU-validated algorithm)
# ---------------------------------------------------------------------------


def _unpack_tokens_low_high(codes: Tensor, ws: int) -> Tensor:
    """Key codes ``[N,H,D,ws//2]`` -> ``[N,H,ws,D]`` (even token in low nibble).

    Mirrors :func:`quantizer.unpack_nibbles_last` + transpose: byte ``j`` holds
    token ``2j`` (low nibble) and ``2j+1`` (high), then channel-major -> token-major.
    """
    low = codes & 0x0F
    high = (codes >> 4) & 0x0F
    toks = torch.stack([low, high], dim=-1).reshape(*codes.shape[:-1], ws)  # [N,H,D,ws]
    return toks.transpose(-2, -1).contiguous()                              # [N,H,ws,D]


def _unpack_channels_low_high(codes: Tensor, D: int) -> Tensor:
    """Value codes ``[N,H,ws,D//2]`` -> ``[N,H,ws,D]`` (even channel in low nibble)."""
    low = codes & 0x0F
    high = (codes >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(*codes.shape[:-1], D)   # [N,H,ws,D]


def _rope_halves(k_pre: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE via the explicit contiguous-half formula the kernel uses.

    ``k_pre`` : ``[..., D]`` pre-RoPE keys. ``cos``/``sin`` : ``[..., D]`` (HF
    layout: first and second half identical). Returns ``[..., D]`` post-RoPE.
    ``rotate_half(x) = [-x2, x1]`` ⇒ out1 = x1·cos1 − x2·sin1, out2 = x2·cos2 + x1·sin2.
    Equivalent to ``transformers.apply_rotary_pos_emb`` — the primitive the
    materialize/oracle path uses (:func:`effective.rotate_key_window`).
    """
    D = k_pre.shape[-1]
    half = D // 2
    x1, x2 = k_pre[..., :half], k_pre[..., half:]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2
    return torch.cat([out1, out2], dim=-1)


def gemv_decode_reference_from_marshalled(
    query: Tensor,
    fp_keys: Tensor,
    fp_values: Tensor,
    marshalled: Optional[Dict[str, Tensor]],
    scaling: float,
    out_dtype: torch.dtype,
) -> Tensor:
    """Pure-torch mirror of the Triton kernel — the CPU-validated algorithm.

    Reproduces the kernel's *exact* arithmetic on the marshalled int4 buffers:
    nibble unpack (key token-parity / value channel-parity), per-channel key grid
    + per-token value grid dequant, contiguous-half RoPE, GQA head grouping, and
    fp32 accumulation. Used to prove — on CPU, against
    :func:`modules.quant.gemv.tiled_gemv_attention` — that the kernel's algorithm
    is correct before it ever runs on a GPU.

    query : ``[H_q, 1, D]``. fp_keys/fp_values : ``[H_kv, T_fp, D]``.
    Returns ``[H_q, 1, D]``.
    """
    h_q, _, D = query.shape
    h_kv, t_fp, _ = fp_keys.shape
    n_rep = h_q // h_kv
    q = query.reshape(h_kv, n_rep, D).to(torch.float32)          # [H_kv, rep, D]

    # --- fp store logits/values (natural order, already post-RoPE) ---
    logits = [torch.einsum("hrd,hsd->hrs", q, fp_keys.to(torch.float32)) * scaling]  # [H_kv,rep,T_fp]
    values = [fp_values.to(torch.float32)]                                          # [H_kv,T_fp,D]

    # --- Q windows: dequant + RoPE per window, exactly as the kernel does ---
    if marshalled is not None:
        ws = marshalled["val_scale"].shape[-1]
        k_codes = _unpack_tokens_low_high(marshalled["key_codes"], ws)     # [N,H,ws,D]
        k_pre = (
            k_codes.to(torch.float32)
            * marshalled["key_scale"].to(torch.float32).unsqueeze(2)       # [N,H,1,D]
            + marshalled["key_zero"].to(torch.float32).unsqueeze(2)
        )                                                                   # [N,H,ws,D]
        cos = marshalled["cos"].to(torch.float32).unsqueeze(1)             # [N,1,ws,D]
        sin = marshalled["sin"].to(torch.float32).unsqueeze(1)
        k_post = _rope_halves(k_pre, cos, sin)                             # [N,H,ws,D]

        v_codes = _unpack_channels_low_high(marshalled["val_codes"], D)    # [N,H,ws,D]
        v = (
            v_codes.to(torch.float32)
            * marshalled["val_scale"].to(torch.float32).unsqueeze(-1)      # [N,H,ws,1]
            + marshalled["val_zero"].to(torch.float32).unsqueeze(-1)
        )                                                                   # [N,H,ws,D]

        N = k_post.shape[0]
        for w in range(N):
            kw = k_post[w]                                                  # [H,ws,D]
            lw = torch.einsum("hrd,hsd->hrs", q, kw) * scaling             # [H_kv,rep,ws]
            logits.append(lw)
            values.append(v[w])                                            # [H,ws,D]

    # --- one softmax over the concatenated key axis (order-invariant) ---
    all_logits = torch.cat(logits, dim=-1)                                 # [H_kv,rep,S]
    all_values = torch.cat(values, dim=1)                                  # [H_kv,S,D]
    p = torch.softmax(all_logits, dim=-1)                                  # [H_kv,rep,S]
    out = torch.einsum("hrs,hsd->hrd", p, all_values)                      # [H_kv,rep,D]
    return out.reshape(h_q, 1, D).to(out_dtype)


# ---------------------------------------------------------------------------
# The Triton kernel + launcher (GPU only; transcribes the mirror above)
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only dev box
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:  # pragma: no cover - requires a GPU to execute

    @triton.jit
    def _gemv_decode_kernel(
        q_ptr,                     # [H_q, D] fp32
        fpk_ptr, fpv_ptr,          # [H_kv, T_fp, D]
        kc_ptr, ks_ptr, kz_ptr,    # key codes [N,H,D,WS//2] u8; scale/zero [N,H,D]
        vc_ptr, vs_ptr, vz_ptr,    # val codes [N,H,WS,D//2] u8; scale/zero [N,H,WS]
        cos_ptr, sin_ptr,          # [N, WS, D]
        out_ptr,                   # [H_q, D]
        T_FP, N_Q,
        scaling,
        stride_qh,
        stride_fk_h, stride_fk_s, stride_fv_h, stride_fv_s,
        stride_kc_n, stride_kc_h, stride_kc_d,
        stride_ks_n, stride_ks_h,
        stride_vc_n, stride_vc_h, stride_vc_s,
        stride_vs_n, stride_vs_h,
        stride_cs_n, stride_cs_s,
        stride_oh,
        N_REP: tl.constexpr,
        D: tl.constexpr, HALF: tl.constexpr,
        WS: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """One program per query head. Serial online softmax over fp + Q tiles.

        Accumulators are kept in fp32. Value channels are accumulated even/odd
        (``acc_lo`` = channels 0,2,4…, ``acc_hi`` = 1,3,5…) to match the int4
        value packing (2 channels/byte, even channel low nibble); the output is
        written back interleaved.
        """
        h = tl.program_id(0)                       # query head
        kv = h // N_REP                            # kv head (GQA)

        offs_d = tl.arange(0, D)
        offs_half = tl.arange(0, HALF)

        # query for this head, split into contiguous halves for RoPE'd dots.
        q = tl.load(q_ptr + h * stride_qh + offs_d).to(tl.float32)   # [D]
        q1 = tl.load(q_ptr + h * stride_qh + offs_half).to(tl.float32)
        q2 = tl.load(q_ptr + h * stride_qh + HALF + offs_half).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc_lo = tl.zeros([HALF], dtype=tl.float32)
        acc_hi = tl.zeros([HALF], dtype=tl.float32)

        # ---- Phase 1: fp store (post-RoPE fp keys, natural order) ----
        for n0 in range(0, T_FP, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T_FP
            k = tl.load(
                fpk_ptr + kv * stride_fk_h + offs_n[:, None] * stride_fk_s + offs_d[None, :],
                mask=mask_n[:, None], other=0.0,
            ).to(tl.float32)                                          # [BLOCK_N, D]
            logit = tl.sum(q[None, :] * k, axis=1) * scaling          # [BLOCK_N]
            logit = tl.where(mask_n, logit, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(logit, axis=0))
            p = tl.exp(logit - m_new)                                 # [BLOCK_N]
            p = tl.where(mask_n, p, 0.0)
            corr = tl.exp(m_i - m_new)

            v_lo = tl.load(
                fpv_ptr + kv * stride_fv_h + offs_n[:, None] * stride_fv_s + (2 * offs_half)[None, :],
                mask=mask_n[:, None], other=0.0,
            ).to(tl.float32)                                          # [BLOCK_N, HALF]
            v_hi = tl.load(
                fpv_ptr + kv * stride_fv_h + offs_n[:, None] * stride_fv_s + (2 * offs_half + 1)[None, :],
                mask=mask_n[:, None], other=0.0,
            ).to(tl.float32)

            l_i = l_i * corr + tl.sum(p, axis=0)
            acc_lo = acc_lo * corr + tl.sum(p[:, None] * v_lo, axis=0)
            acc_hi = acc_hi * corr + tl.sum(p[:, None] * v_hi, axis=0)
            m_i = m_new

        # ---- Phase 2: Q tier, one window (= one tile = one scale group) ----
        offs_w = tl.arange(0, BLOCK_W)
        mask_w = offs_w < WS
        for w in range(0, N_Q):
            # key codes: channel-major [N,H,D,WS//2]; byte(d,t) at ...+ d*stride_kc_d + t//2
            byte = tl.load(
                kc_ptr + w * stride_kc_n + kv * stride_kc_h
                + offs_d[None, :] * stride_kc_d + (offs_w[:, None] // 2),
                mask=mask_w[:, None], other=0,
            )                                                          # [BLOCK_W, D] u8
            parity = (offs_w % 2)[:, None]                             # [BLOCK_W,1]
            nib = tl.where(parity == 0, byte & 0x0F, (byte >> 4) & 0x0F).to(tl.float32)
            kscale = tl.load(ks_ptr + w * stride_ks_n + kv * stride_ks_h + offs_d).to(tl.float32)
            kzero = tl.load(kz_ptr + w * stride_ks_n + kv * stride_ks_h + offs_d).to(tl.float32)
            k_pre = nib * kscale[None, :] + kzero[None, :]             # [BLOCK_W, D] pre-RoPE

            c1 = tl.load(cos_ptr + w * stride_cs_n + offs_w[:, None] * stride_cs_s + offs_half[None, :]).to(tl.float32)
            c2 = tl.load(cos_ptr + w * stride_cs_n + offs_w[:, None] * stride_cs_s + HALF + offs_half[None, :]).to(tl.float32)
            s1 = tl.load(sin_ptr + w * stride_cs_n + offs_w[:, None] * stride_cs_s + offs_half[None, :]).to(tl.float32)
            s2 = tl.load(sin_ptr + w * stride_cs_n + offs_w[:, None] * stride_cs_s + HALF + offs_half[None, :]).to(tl.float32)
            k1 = k_pre[:, 0:HALF]
            k2 = k_pre[:, HALF:D]
            kr1 = k1 * c1 - k2 * s1                                    # [BLOCK_W, HALF]
            kr2 = k2 * c2 + k1 * s2
            logit = (tl.sum(q1[None, :] * kr1, axis=1) + tl.sum(q2[None, :] * kr2, axis=1)) * scaling
            logit = tl.where(mask_w, logit, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(logit, axis=0))
            p = tl.exp(logit - m_new)
            p = tl.where(mask_w, p, 0.0)
            corr = tl.exp(m_i - m_new)

            # value codes: token-major [N,H,WS,D//2], 2 channels/byte
            vbyte = tl.load(
                vc_ptr + w * stride_vc_n + kv * stride_vc_h
                + offs_w[:, None] * stride_vc_s + tl.arange(0, HALF)[None, :],
                mask=mask_w[:, None], other=0,
            )                                                          # [BLOCK_W, HALF] u8
            v_lo = (vbyte & 0x0F).to(tl.float32)                       # even channels
            v_hi = ((vbyte >> 4) & 0x0F).to(tl.float32)                # odd channels
            vscale = tl.load(vs_ptr + w * stride_vs_n + kv * stride_vs_h + offs_w, mask=mask_w, other=0.0).to(tl.float32)
            vzero = tl.load(vz_ptr + w * stride_vs_n + kv * stride_vs_h + offs_w, mask=mask_w, other=0.0).to(tl.float32)
            v_lo = v_lo * vscale[:, None] + vzero[:, None]
            v_hi = v_hi * vscale[:, None] + vzero[:, None]

            l_i = l_i * corr + tl.sum(p, axis=0)
            acc_lo = acc_lo * corr + tl.sum(p[:, None] * v_lo, axis=0)
            acc_hi = acc_hi * corr + tl.sum(p[:, None] * v_hi, axis=0)
            m_i = m_new

        o_lo = acc_lo / l_i
        o_hi = acc_hi / l_i
        tl.store(out_ptr + h * stride_oh + 2 * offs_half, o_lo)
        tl.store(out_ptr + h * stride_oh + 2 * offs_half + 1, o_hi)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def gemv_decode_triton(
    query: Tensor,
    fp_keys: Tensor,
    fp_values: Tensor,
    store,
    rope_module: torch.nn.Module,
    scaling: Optional[float] = None,
    out_dtype: Optional[torch.dtype] = None,
    block_n: int = 64,
) -> Tensor:
    """Launch the fused Triton decode kernel. Requires CUDA + triton.

    Same contract as :func:`modules.quant.gemv.tiled_gemv_attention`:
    ``query`` ``[H_q,1,D]`` or ``[1,H_q,1,D]``, ``fp_keys/fp_values`` ``[H_kv,T_fp,D]``,
    returns the matching shape. Consumes the int4 Q tier tile-by-tile — the fp16
    blow-up stays in registers (never written to HBM).
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not importable; use the reference path.")

    had_batch = query.dim() == 4
    q4 = query[0] if had_batch else query
    if q4.dim() != 3 or q4.shape[1] != 1:
        raise NotImplementedError("gemv_decode_triton is the decode (single-query) path.")

    h_q, _, D = q4.shape
    h_kv, t_fp, _ = fp_keys.shape
    n_rep = h_q // h_kv
    if scaling is None:
        scaling = D ** -0.5
    if out_dtype is None:
        out_dtype = query.dtype
    device = q4.device

    q = q4.reshape(h_q, D).contiguous().to(torch.float32)
    fp_keys = fp_keys.contiguous()
    fp_values = fp_values.contiguous()

    marshalled = _marshal_q_windows(store, rope_module, fp_keys) if store is not None else None
    n_q = 0
    ws = store.window_size if store is not None else 2
    if marshalled is not None:
        for t in marshalled:
            marshalled[t] = marshalled[t].contiguous().to(device)
        n_q = marshalled["key_codes"].shape[0]
        ws = marshalled["val_scale"].shape[-1]

    out = torch.empty((h_q, D), device=device, dtype=torch.float32)

    # Zero-strided placeholders so the kernel's Q-phase args are always valid,
    # even when the Q tier is empty (N_Q == 0 ⇒ the phase loop never runs).
    def _mk(shape, dt=torch.uint8):
        return torch.zeros(shape, device=device, dtype=dt)

    kc = marshalled["key_codes"] if marshalled else _mk((1, h_kv, D, max(ws // 2, 1)))
    ks = marshalled["key_scale"] if marshalled else _mk((1, h_kv, D), torch.float16)
    kz = marshalled["key_zero"] if marshalled else _mk((1, h_kv, D), torch.float16)
    vc = marshalled["val_codes"] if marshalled else _mk((1, h_kv, ws, max(D // 2, 1)))
    vs = marshalled["val_scale"] if marshalled else _mk((1, h_kv, ws), torch.float16)
    vz = marshalled["val_zero"] if marshalled else _mk((1, h_kv, ws), torch.float16)
    cos = marshalled["cos"] if marshalled else _mk((1, ws, D), torch.float32)
    sin = marshalled["sin"] if marshalled else _mk((1, ws, D), torch.float32)

    grid = (h_q,)
    _gemv_decode_kernel[grid](
        q, fp_keys, fp_values,
        kc, ks, kz, vc, vs, vz, cos, sin,
        out,
        t_fp, n_q,
        float(scaling),
        q.stride(0),
        fp_keys.stride(0), fp_keys.stride(1), fp_values.stride(0), fp_values.stride(1),
        kc.stride(0), kc.stride(1), kc.stride(2),
        ks.stride(0), ks.stride(1),
        vc.stride(0), vc.stride(1), vc.stride(2),
        vs.stride(0), vs.stride(1),
        cos.stride(0), cos.stride(1),
        out.stride(0),
        N_REP=n_rep,
        D=D, HALF=D // 2,
        WS=ws, BLOCK_W=_next_pow2(ws), BLOCK_N=block_n,
    )

    out = out.reshape(h_q, 1, D).to(out_dtype)
    return out.unsqueeze(0) if had_batch else out
