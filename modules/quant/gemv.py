"""Tiled GEMV decode attention — dequant-inside-attention, one window at a time.

This is the Phase-2 read path of design.md §11 realized in portable PyTorch: a
streaming (online-softmax) attention over the two-tier KV cache for a **single
decode query**, where the int4 Q tier is consumed **one window at a time** and
the full effective K/V is **never materialized**.

Why this exists
---------------
The Phase-1 read path (:func:`modules.quant.effective.materialize_effective_kv`)
dequantizes the entire Q tier to fp16, glues it to the fp store, and hands the
whole ``[H_kv, T_total, D]`` tensor to the standard attention op. That transient
fp16 blow-up of the quantized portion is the dominant Phase-1 cost (~4× the int4
read; design §11). The tiled path removes it: the design principle we hold to is

    *never materialize the full fp copy of the quantized portion — dequantize a
    Q window, consume it (score + weight into the running softmax), and drop it.*

So the largest fp copy of the Q tier ever resident is **one window**, regardless
of how many Q windows the cache holds.

Correctness
-----------
Attention over a single query is ``softmax(q·Kᵀ · scaling) · V`` — a reduction
over the key axis. The online-softmax recurrence (Milakov & Gimelshein 2018;
the flash-attention accumulator) computes exactly that reduction incrementally,
combining tiles in **any order** with a running ``(max, denom, weighted-value)``
state. Attention is permutation-invariant over keys (each key already carries
its position via RoPE), so tile order is irrelevant to the result — we do not
need the chronological interleave here (that only ever mattered for the *window
scorer*, which chunks the physical key axis; design §5, §8).

Accumulators run in **fp32** for stability, independent of the KV dtype. Each
tile's keys/values are cast to fp32 for the MACs and the result is cast back to
the query dtype. Against a full fp32 softmax over the same (dequantized, RoPE'd)
keys the output matches to fp32 reduction-order tolerance.

Scope
-----
Decode only (``T_q == 1`` query tokens) — the GEMV regime the design targets;
prefill stays on the materialize path (§11). B = 1 (v1; design §10). GQA is
handled without expanding the KV heads: query heads are viewed as
``(H_kv, n_rep)`` and the tile MAC broadcasts over ``n_rep`` — no ``repeat_kv``
copy, exactly as the flash score hook does.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .effective import rotate_key_window


class _OnlineSoftmax:
    """Running flash-attention accumulator for one decode query (fp32, B = 1).

    State per ``(H_kv, n_rep)`` query slot:

    - ``m``   : running max logit                      ``[H_kv, n_rep]``
    - ``l``   : running softmax denominator (Σ exp)     ``[H_kv, n_rep]``
    - ``acc`` : running Σ (exp · value)                 ``[H_kv, n_rep, D]``

    :meth:`absorb` folds one key/value tile into the state; :meth:`result`
    finalizes ``acc / l``. Empty tiles are ignored. The ``m = -inf`` init makes
    the first tile fall out cleanly (``exp(-inf) = 0``).
    """

    def __init__(self, h_kv: int, n_rep: int, head_dim: int, device) -> None:
        f32 = torch.float32
        self.m = torch.full((h_kv, n_rep), float("-inf"), device=device, dtype=f32)
        self.l = torch.zeros((h_kv, n_rep), device=device, dtype=f32)
        self.acc = torch.zeros((h_kv, n_rep, head_dim), device=device, dtype=f32)

    def absorb(self, q32: Tensor, keys: Tensor, values: Tensor, scaling: float) -> None:
        """Fold one tile into the running softmax.

        Parameters
        ----------
        q32 : ``[H_kv, n_rep, D]`` fp32 — the (post-RoPE) query, head-grouped.
        keys, values : ``[H_kv, n_t, D]`` — one tile's post-RoPE keys / values.
            Cast to fp32 here; ``n_t`` may be any positive length (a window, the
            whole fp store, or an fp sub-chunk).
        scaling : float — the attention softmax scale (usually ``D**-0.5``).
        """
        n_t = keys.shape[1]
        if n_t == 0:
            return
        kf = keys.to(torch.float32)
        vf = values.to(torch.float32)

        # logits[h, r, n] = scaling · Σ_d q[h,r,d] · k[h,n,d]
        logits = torch.einsum("hrd,hnd->hrn", q32, kf) * scaling  # [H_kv, n_rep, n_t]

        tile_max = logits.amax(dim=-1)                       # [H_kv, n_rep]
        m_new = torch.maximum(self.m, tile_max)              # [H_kv, n_rep]
        # exp(m_old - m_new): 0 on the first tile (m_old = -inf), in (0, 1] after.
        corr = torch.exp(self.m - m_new)                     # [H_kv, n_rep]
        p = torch.exp(logits - m_new.unsqueeze(-1))          # [H_kv, n_rep, n_t]

        self.l = self.l * corr + p.sum(dim=-1)
        self.acc = self.acc * corr.unsqueeze(-1) + torch.einsum("hrn,hnd->hrd", p, vf)
        self.m = m_new

    def result(self, out_dtype: torch.dtype) -> Tensor:
        """Return the finalized attention output ``[H_kv, n_rep, D]`` in ``out_dtype``.

        A query slot that saw no keys (``l == 0``) yields zeros rather than NaN.
        """
        denom = self.l.clamp_min(torch.finfo(torch.float32).tiny)
        out = self.acc / denom.unsqueeze(-1)
        return out.to(out_dtype)


def tiled_gemv_attention(
    query: Tensor,
    fp_keys: Tensor,
    fp_values: Tensor,
    store,
    rope_module: torch.nn.Module,
    scaling: Optional[float] = None,
    fp_tile_size: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Decode attention over the two-tier cache, Q tier consumed window-by-window.

    Computes ``softmax(q·Kᵀ·scaling)·V`` for one decode query, streaming the keys
    as tiles: the fp store (already fp16, cheap) plus each active Q window
    (dequantized + RoPE-stamped on the fly, then discarded). The full effective
    K/V is never built (design §11, Phase 2). Numerically equal — to fp32
    reduction-order tolerance — to a full softmax over
    :func:`materialize_effective_kv`'s output.

    Parameters
    ----------
    query : ``[H_q, 1, D]`` or ``[1, H_q, 1, D]``
        The post-RoPE decode query for one row. ``T_q`` must be 1 (decode).
    fp_keys, fp_values : ``[H_kv, T_fp, D]``
        The fp store for row 0 (post-RoPE keys). ``T_fp`` includes sink + local
        + any fp survivor windows; sink needs no special handling here (it is
        just more fp keys — sink stripping is a *scorer* concern, not attention).
    store : QuantizedStore or None
        The Q tier. ``None`` or empty ⇒ plain fp attention over the fp store.
    rope_module : nn.Module
        The model's rotary embedding, used to RoPE each Q window at its frozen
        original positions (same primitive the materialize path uses).
    scaling : float, optional
        Softmax scale. Defaults to ``D ** -0.5``.
    fp_tile_size : int, optional
        If given, the fp store is streamed in chunks of this many tokens (bounds
        the per-tile logit block). Default ``None`` ⇒ the fp store is one tile
        (it is already resident fp16; there is nothing to save by splitting it,
        but the option keeps the kernel uniformly tiled for large ``T_fp``).
    out_dtype : torch.dtype, optional
        Output dtype. Defaults to ``query``'s dtype.

    Returns
    -------
    attn : ``[H_q, 1, D]`` (or ``[1, H_q, 1, D]`` if ``query`` had a batch axis).
    """
    had_batch = query.dim() == 4
    if had_batch:
        if query.shape[0] != 1:
            raise NotImplementedError(
                f"tiled_gemv_attention is B = 1 only (v1); got batch {query.shape[0]}."
            )
        q4 = query[0]
    else:
        q4 = query
    if q4.dim() != 3:
        raise ValueError(
            f"query must be [H_q, 1, D] or [1, H_q, 1, D], got {tuple(query.shape)}"
        )
    if q4.shape[1] != 1:
        raise NotImplementedError(
            "tiled_gemv_attention is the decode (single-query) path; got "
            f"{q4.shape[1]} query tokens. Prefill stays on the materialize path "
            "(design.md §11)."
        )

    h_q, _, head_dim = q4.shape
    h_kv = fp_keys.shape[0]
    if h_q % h_kv != 0:
        raise ValueError(
            f"H_q ({h_q}) must be a multiple of H_kv ({h_kv}) for GQA grouping."
        )
    n_rep = h_q // h_kv
    if scaling is None:
        scaling = head_dim ** -0.5
    if out_dtype is None:
        out_dtype = query.dtype
    device = q4.device

    # Query head-grouped to [H_kv, n_rep, D], fp32 for the MACs. The kv head that
    # owns query heads [g*n_rep : (g+1)*n_rep] is contiguous in the standard HF
    # head layout, so a plain reshape lines them up (matches the flash hook).
    q32 = q4.reshape(h_kv, n_rep, head_dim).to(torch.float32)

    # Dtype the Q tier dequants to before RoPE — match the fp store so a Q
    # window's post-RoPE key is bit-identical to the materialize path's before
    # the shared fp32 cast (keeps the two paths in exact agreement).
    q_deq_dtype = fp_keys.dtype

    state = _OnlineSoftmax(h_kv, n_rep, head_dim, device)

    # --- Tier 1: the fp store, streamed in fp_tile_size chunks (or one tile) ---
    t_fp = fp_keys.shape[1]
    if fp_tile_size is None or fp_tile_size <= 0:
        step = t_fp
    else:
        step = fp_tile_size
    for start in range(0, t_fp, step):
        end = min(start + step, t_fp)
        state.absorb(q32, fp_keys[:, start:end], fp_values[:, start:end], scaling)

    # --- Tier 2: the int4 Q store, ONE window at a time (never the whole tier) --
    if store is not None and store.num_active_windows > 0:
        for _wid, k_pre, v_win, pos in store.iter_active_windows(out_dtype=q_deq_dtype):
            k_post = rotate_key_window(k_pre, pos, rope_module)  # [H_kv, ws, D]
            state.absorb(q32, k_post, v_win, scaling)
            # k_pre / k_post / v_win fall out of scope here — the fp blow-up of
            # this window is released before the next window is dequantized.

    out = state.result(out_dtype).reshape(h_q, 1, head_dim)  # [H_q, 1, D]
    if had_batch:
        return out.unsqueeze(0)  # [1, H_q, 1, D]
    return out


def gemv_decode(
    query: Tensor,
    fp_keys: Tensor,
    fp_values: Tensor,
    store,
    rope_module: torch.nn.Module,
    scaling: Optional[float] = None,
    out_dtype: Optional[torch.dtype] = None,
    prefer_triton: bool = True,
) -> Tensor:
    """Decode attention dispatcher: fused Triton kernel on GPU, else the reference.

    Identical contract and result (to fp32 tolerance) as
    :func:`tiled_gemv_attention`. When ``prefer_triton`` and a CUDA + triton path
    is available, the fused Phase-2 kernel runs (int4 codes consumed tile-by-tile,
    fp16 blow-up kept in registers — never written to HBM). Everywhere else —
    including this CPU dev box — it transparently falls back to the portable
    :func:`tiled_gemv_attention` reference, so callers need no GPU branch.
    """
    if prefer_triton:
        try:
            from .gemv_triton import gemv_decode_triton, triton_available

            if triton_available() and query.is_cuda:
                return gemv_decode_triton(
                    query, fp_keys, fp_values, store, rope_module,
                    scaling=scaling, out_dtype=out_dtype,
                )
        except Exception:
            # Any import/launch problem ⇒ fall back to the exact reference path.
            pass
    return tiled_gemv_attention(
        query, fp_keys, fp_values, store, rope_module,
        scaling=scaling, out_dtype=out_dtype,
    )
