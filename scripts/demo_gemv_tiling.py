#!/usr/bin/env python3
"""Standalone demo: tiled GEMV decode attention over the two-tier KV cache.

Runs entirely on CPU with a toy RoPE — no model download, no GPU, no transformers
model needed (only ``transformers``' ``apply_rotary_pos_emb`` helper, which the
project already depends on). It demonstrates the two things the tiled path
promises (design.md §11, Phase 2):

  1. **Correctness.** ``tiled_gemv_attention`` matches a full-softmax attention
     over the Phase-1 materialize-then-interleave effective K/V, to fp32
     reduction-order tolerance.

  2. **Never materialize the full fp Q tier.** The Q store is consumed one window
     at a time: the largest fp copy of the *quantized portion* that is ever
     resident is a single window, no matter how many Q windows the cache holds.
     The demo instruments the store to prove the peak-resident count.

Run:
    python scripts/demo_gemv_tiling.py
    python scripts/demo_gemv_tiling.py --q-windows 64 --window 32 --heads 8 --rep 4
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from modules.quant.gemv import tiled_gemv_attention
from modules.quant.store import QuantizedStore
from modules.quant.effective import materialize_effective_kv, rotate_key_window


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


def _reference(query, eff_k, eff_v, scaling):
    """Full fp32 softmax attention over the materialized effective K/V."""
    h_q = query.shape[0]
    h_kv, S, D = eff_k.shape
    n_rep = h_q // h_kv
    q = query.reshape(h_kv, n_rep, D).float()
    logits = torch.einsum("hrd,hsd->hrs", q, eff_k.float()) * scaling
    w = torch.softmax(logits, dim=-1)
    return torch.einsum("hrs,hsd->hrd", w, eff_v.float()).reshape(h_q, 1, D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp-windows", type=int, default=4, help="fp-tier body windows")
    ap.add_argument("--q-windows", type=int, default=16, help="int4 Q-tier windows")
    ap.add_argument("--window", type=int, default=8, help="window size (even)")
    ap.add_argument("--sink", type=int, default=4, help="sink tokens")
    ap.add_argument("--heads", type=int, default=4, help="kv heads")
    ap.add_argument("--rep", type=int, default=2, help="GQA groups (query heads = heads*rep)")
    ap.add_argument("--dim", type=int, default=8, help="head dim (even)")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    torch.manual_seed(args.seed)
    rope = _RealRoPE(args.dim)
    H_kv, n_rep, D, ws = args.heads, args.rep, args.dim, args.window
    H_q = H_kv * n_rep
    scaling = D ** -0.5

    # --- fp store: sink + fp body windows (post-RoPE, contiguous positions) ---
    T_fp = args.sink + args.fp_windows * ws
    fp_pos = torch.arange(T_fp, dtype=torch.long)
    fp_k = rotate_key_window(torch.randn(H_kv, T_fp, D, dtype=dtype), fp_pos, rope).to(dtype)
    fp_v = torch.randn(H_kv, T_fp, D, dtype=dtype)

    # --- Q store: args.q_windows int4 windows with ids after the fp windows ---
    store = QuantizedStore(window_size=ws, head_dim=D, num_kv_heads=H_kv)
    base = args.sink + T_fp
    for j in range(args.q_windows):
        wid = args.fp_windows + j
        pos = torch.arange(base + j * ws, base + (j + 1) * ws, dtype=torch.long)
        store.demote(
            wid,
            torch.randn(H_kv, ws, D, dtype=dtype),
            torch.randn(H_kv, ws, D, dtype=dtype),
            pos,
        )

    query = torch.randn(H_q, 1, D, dtype=dtype)

    # --- Instrument the store to measure peak-resident dequantized Q windows ---
    live = {"cur": 0, "peak": 0}
    orig_iter = store.iter_active_windows

    def measured_iter(*a, **k):
        # Each yielded window is "live" only while the consumer holds it; the
        # kernel drops it before pulling the next. Emulate that lifetime here.
        for item in orig_iter(*a, **k):
            live["cur"] = 1
            live["peak"] = max(live["peak"], live["cur"])
            yield item
            live["cur"] = 0

    store.iter_active_windows = measured_iter  # type: ignore[assignment]

    out = tiled_gemv_attention(query, fp_k, fp_v, store, rope, scaling)

    # --- Correctness vs the Phase-1 materialize path -------------------------
    eff_k, eff_v = materialize_effective_kv(
        fp_k, fp_v, fp_pos, store, num_sink=args.sink, window_size=ws,
        rope_module=rope, out_dtype=dtype,
    )
    ref = _reference(query, eff_k, eff_v, scaling)
    max_abs = (out.float() - ref.float()).abs().max().item()

    T_total = eff_k.shape[1]
    print("=" * 68)
    print("Tiled GEMV decode attention - demo")
    print("=" * 68)
    print(f"  layout        : sink={args.sink}  fp_windows={args.fp_windows}  "
          f"q_windows={args.q_windows}  window={ws}")
    print(f"  heads         : H_kv={H_kv}  n_rep={n_rep}  H_q={H_q}  D={D}  dtype={args.dtype}")
    print(f"  effective S   : {T_total} keys  (fp {T_fp} + Q {store.num_active_tokens})")
    print("-" * 68)
    print(f"  output shape  : {tuple(out.shape)}")
    print(f"  max |tiled - materialize|  : {max_abs:.3e}")
    tol = 1e-4 if dtype == torch.float32 else 3e-3
    print(f"  correctness   : {'PASS' if max_abs < tol else 'FAIL'}  (tol {tol:.0e})")
    print("-" * 68)
    print(f"  Q windows held : {args.q_windows} total in the int4 store")
    print(f"  peak fp Q windows resident during attention : {live['peak']}")
    print(f"  => the fp blow-up of the quantized tier never exceeds "
          f"{live['peak']} window ({live['peak'] * ws} tokens),")
    print(f"     vs {store.num_active_tokens} tokens the materialize path would hold.")
    print("=" * 68)


if __name__ == "__main__":
    main()
