"""Triton fused GEMV decode kernel — validation (design.md §11, Phase 2).

The kernel itself needs a GPU (skipped here on the CPU dev box). What CAN be
validated on CPU is the thing that carries the correctness risk: the kernel's
**exact arithmetic** — nibble parity, per-channel key grid / per-token value grid
dequant, contiguous-half RoPE, even/odd value unpack, GQA head mapping. That
logic is transcribed verbatim into the Triton kernel from the pure-torch
``gemv_decode_reference_from_marshalled`` mirror, so proving the mirror equals the
streaming oracle (``tiled_gemv_attention``) validates the algorithm end to end.
Only the Triton syntax/launch remains for the GPU-gated test.

Run the GPU test on a CUDA box:  ``pytest -m gpu tests/test_gemv_triton.py``
"""

from __future__ import annotations

import pytest
import torch

from modules.quant.gemv import gemv_decode, tiled_gemv_attention
from modules.quant.store import QuantizedStore
from modules.quant.effective import rotate_key_window, _rope_cos_sin
from modules.quant.gemv_triton import (
    _marshal_q_windows,
    gemv_decode_reference_from_marshalled,
    triton_available,
)


class _RealRoPE(torch.nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        B = position_ids.shape[0]
        inv = self.inv_freq[None, :, None].float().expand(B, -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _build_two_tier(
    n_fp_windows=3, n_q_windows=4, ws=2, num_sink=2, H_kv=2, n_rep=1, D=4,
    dtype=torch.float32, seed=0,
):
    torch.manual_seed(seed)
    rope = _RealRoPE(D)
    H_q = H_kv * n_rep
    T_fp = num_sink + n_fp_windows * ws
    fp_pos = torch.arange(T_fp, dtype=torch.long)
    fp_k = rotate_key_window(torch.randn(H_kv, T_fp, D, dtype=dtype), fp_pos, rope).to(dtype)
    fp_v = torch.randn(H_kv, T_fp, D, dtype=dtype)
    store = QuantizedStore(window_size=ws, head_dim=D, num_kv_heads=H_kv)
    base = num_sink + T_fp
    for j in range(n_q_windows):
        wid = n_fp_windows + j
        pos = torch.arange(base + j * ws, base + (j + 1) * ws, dtype=torch.long)
        store.demote(
            wid, torch.randn(H_kv, ws, D, dtype=dtype),
            torch.randn(H_kv, ws, D, dtype=dtype), pos,
        )
    return dict(
        rope=rope, query=torch.randn(H_q, 1, D, dtype=dtype), scaling=D ** -0.5,
        store=store, fp_k=fp_k, fp_v=fp_v, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# 1. The mirror == the streaming oracle (validates the kernel's ALGORITHM on CPU)
# ---------------------------------------------------------------------------


def _mirror(s):
    marshalled = _marshal_q_windows(s["store"], s["rope"], s["fp_k"])
    return gemv_decode_reference_from_marshalled(
        s["query"], s["fp_k"], s["fp_v"], marshalled, s["scaling"], s["dtype"]
    )


def test_mirror_matches_oracle_two_tier():
    s = _build_two_tier(dtype=torch.float32)
    ref = tiled_gemv_attention(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out = _mirror(s)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


@pytest.mark.parametrize("n_rep", [1, 2, 4])
def test_mirror_matches_oracle_gqa(n_rep):
    s = _build_two_tier(H_kv=2, n_rep=n_rep, dtype=torch.float32)
    ref = tiled_gemv_attention(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out = _mirror(s)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


def test_mirror_matches_oracle_empty_q_tier():
    s = _build_two_tier(n_q_windows=0, dtype=torch.float32)
    assert _marshal_q_windows(s["store"], s["rope"], s["fp_k"]) is None
    ref = tiled_gemv_attention(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out = _mirror(s)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_mirror_matches_oracle_fp16():
    s = _build_two_tier(dtype=torch.float16)
    ref = tiled_gemv_attention(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out = _mirror(s)
    assert torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("seed", range(8))
def test_mirror_fuzz_matches_oracle(seed):
    s = _build_two_tier(
        n_fp_windows=1 + seed % 4, n_q_windows=1 + (seed * 3) % 5,
        ws=2 + 2 * (seed % 3), num_sink=seed % 3, H_kv=1 + seed % 2,
        n_rep=1 + seed % 3, D=4 + 4 * (seed % 2), dtype=torch.float32, seed=seed,
    )
    ref = tiled_gemv_attention(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out = _mirror(s)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


# ---------------------------------------------------------------------------
# 2. Marshalling: shapes + cos/sin correctness
# ---------------------------------------------------------------------------


def test_marshal_shapes_and_cos_sin():
    s = _build_two_tier(n_q_windows=3, ws=4, H_kv=2, D=8, dtype=torch.float32)
    m = _marshal_q_windows(s["store"], s["rope"], s["fp_k"])
    N, H, D, wshalf = m["key_codes"].shape
    assert (N, H, D, wshalf) == (3, 2, 8, 2)               # ws//2 = 2
    assert m["val_codes"].shape == (3, 2, 4, 4)            # [N,H,ws,D//2]
    assert m["key_scale"].shape == (3, 2, 8)
    assert m["val_scale"].shape == (3, 2, 4)
    assert m["cos"].shape == (3, 4, 8) == m["sin"].shape
    # cos/sin equal a direct rope call at each window's frozen positions.
    for i, e in enumerate(s["store"].ledger.active_entries()):
        cos, sin = _rope_cos_sin(s["rope"], s["fp_k"], e.position_range)
        assert torch.allclose(m["cos"][i], cos[0])
        assert torch.allclose(m["sin"][i], sin[0])


# ---------------------------------------------------------------------------
# 3. Dispatcher: CPU falls back to the exact reference
# ---------------------------------------------------------------------------


def test_dispatcher_falls_back_to_reference_on_cpu():
    s = _build_two_tier(dtype=torch.float32)
    ref = tiled_gemv_attention(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out = gemv_decode(s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    assert torch.equal(out, ref)  # CPU ⇒ identical to the reference path


# ---------------------------------------------------------------------------
# 4. The real Triton kernel — GPU only (skipped on the CPU dev box)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not triton_available(), reason="requires CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n_rep", [1, 4])
def test_triton_kernel_matches_oracle(dtype, n_rep):
    from modules.quant.gemv_triton import gemv_decode_triton

    s = _build_two_tier(n_fp_windows=3, n_q_windows=6, ws=8, H_kv=2, n_rep=n_rep, D=16,
                        dtype=dtype, seed=3)
    dev = "cuda"
    q = s["query"].to(dev)
    fp_k = s["fp_k"].to(dev)
    fp_v = s["fp_v"].to(dev)
    rope = s["rope"].to(dev)
    # move the store's ledger tensors to the GPU
    for e in s["store"].ledger.active_entries():
        e.key_codes = e.key_codes.to(dev); e.key_scale = e.key_scale.to(dev)
        e.key_zero = e.key_zero.to(dev); e.val_codes = e.val_codes.to(dev)
        e.val_scale = e.val_scale.to(dev); e.val_zero = e.val_zero.to(dev)
        e.position_range = e.position_range.to(dev)

    ref = tiled_gemv_attention(q, fp_k, fp_v, s["store"], rope, s["scaling"])
    out = gemv_decode_triton(q, fp_k, fp_v, s["store"], rope, s["scaling"])
    atol = 1e-3 if dtype == torch.float32 else 5e-3
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=atol), \
        (out.float() - ref.float()).abs().max()
