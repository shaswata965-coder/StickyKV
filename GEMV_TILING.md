# Tiled GEMV decode attention (design.md §11, Phase 2)

Streaming, online-softmax attention over the two-tier KV cache that **consumes
the int4 Q tier one window at a time and never materializes the full effective
K/V**. This is the Phase-2 read path of `design.md` §11.

It ships in two layers:

- a **portable PyTorch reference** (`tiled_gemv_attention`) — fully CPU-testable,
  the correctness anchor, and the CPU fallback; and
- the **fused Triton GPU kernel** (`gemv_decode_triton`) — the design's actual
  Phase-2 endpoint: int4 codes loaded tile-by-tile, dequantized + RoPE'd **in
  registers**, MAC'd against the query, so the fp16 blow-up is never written to
  HBM (only int4 codes are read from HBM).

A dispatcher (`gemv_decode`) picks the Triton kernel when CUDA + triton are
present and transparently falls back to the reference everywhere else, so callers
need no GPU branch.

> **Validation status.** The reference and the kernel's full arithmetic are
> CPU-validated (see §4b). The Triton kernel's on-GPU execution is **not yet
> validated** — this dev box is CPU-only (no CUDA, no triton). It is authored to
> transcribe the CPU-validated mirror line-for-line; run `pytest -m gpu` on a GPU
> box to close the gap (§6).

---

## 1. The design principle it enforces

The Phase-1 read path
([`materialize_effective_kv`](modules/quant/effective.py)) dequantizes the
**entire** Q tier to fp16, glues it to the fp store, and hands the whole
`[H_kv, T_total, D]` tensor to the standard attention op. That transient fp16
blow-up of the quantized portion is the dominant Phase-1 cost (~4× the int4 read).

The tiled path removes it. The invariant it holds to:

> **Never materialize the full fp copy of the quantized portion.** Dequantize a
> Q window, consume it (score + weight into the running softmax), and drop it
> before touching the next window. Only int4 codes live in memory; the fp16
> blow-up happens window-by-window, on demand, and is discarded immediately.

So the largest fp copy of the Q tier ever resident is **one window**, regardless
of how many Q windows the cache holds. The demo prints this directly:

```
  Q windows held : 64 total in the int4 store
  peak fp Q windows resident during attention : 1
  => the fp blow-up of the quantized tier never exceeds 1 window (32 tokens),
     vs 2048 tokens the materialize path would hold.
```

---

## 2. Why it is correct

Attention over a single decode query is `softmax(q·Kᵀ · scaling) · V` — a
reduction over the key axis. The **online-softmax** recurrence (the
flash-attention accumulator) computes exactly that reduction incrementally,
combining tiles in **any order** via a running `(max, denom, weighted-value)`
state. Attention is permutation-invariant over keys (each key already carries its
position through RoPE), so tile order is irrelevant to the result — the
chronological interleave is **not** needed here (it only ever mattered for the
window *scorer*, which chunks the physical key axis; `design.md` §5, §8).

Each Q window is dequantized and RoPE-stamped at its **frozen original
positions** using the very same primitive the materialize path uses
([`rotate_key_window`](modules/quant/effective.py)), so a window's post-RoPE key
is bit-identical across the two paths. Accumulators run in **fp32** for
stability; against a full fp32 softmax over the materialized effective K/V the
tiled output matches to fp32 reduction-order tolerance.

---

## 3. What was added

| File | Change |
|---|---|
| [`modules/quant/gemv.py`](modules/quant/gemv.py) | **new** — `tiled_gemv_attention` + `_OnlineSoftmax` (the PyTorch reference) and `gemv_decode` (the Triton/reference dispatcher). |
| [`modules/quant/gemv_triton.py`](modules/quant/gemv_triton.py) | **new** — the fused Triton kernel `_gemv_decode_kernel` + launcher `gemv_decode_triton`; host-side int4 marshalling `_marshal_q_windows`; and `gemv_decode_reference_from_marshalled`, a pure-torch **mirror** of the kernel's exact arithmetic (the CPU validation surface). |
| [`modules/quant/store.py`](modules/quant/store.py) | **new** `QuantizedStore.iter_active_windows()` — lazy, one-window-at-a-time dequant (the read primitive that enforces the invariant). |
| [`modules/quant/__init__.py`](modules/quant/__init__.py) | export `tiled_gemv_attention`, `gemv_decode`. |
| [`modules/windowed_cache/cache.py`](modules/windowed_cache/cache.py) · [`…eager_cache/cache.py`](modules/windowed_eager_cache/cache.py) | **new** `WindowedCache.decode_attention()` — the cache entry point, routed through `gemv_decode` (mirrored into both backends; the twins stay byte-identical apart from their docstring). |
| [`tests/test_gemv.py`](tests/test_gemv.py) | **new** — 22 tests (reference equivalence, GQA, fp16, fuzz, the one-window invariant, order invariance, cache integration for both backends). |
| [`tests/test_gemv_triton.py`](tests/test_gemv_triton.py) | **new** — 16 CPU tests (mirror == oracle, marshalling, dispatcher fallback) + 4 GPU-gated tests (the real kernel vs the oracle). |
| [`scripts/demo_gemv_tiling.py`](scripts/demo_gemv_tiling.py) | **new** — standalone CPU demo. |
| [`conftest.py`](conftest.py) | **new** — registers the `gpu` marker. |

Nothing in the Phase-1 path changed; `materialize_effective_kv` remains the
prefill path and the correctness anchor.

### How the kernel is de-risked without a GPU

The kernel's correctness risk is all in its *arithmetic* — nibble parity,
per-channel key grid / per-token value grid dequant, contiguous-half RoPE,
even/odd value unpack, GQA head mapping. That logic lives once in
`gemv_decode_reference_from_marshalled` (pure torch, runs on CPU) and is asserted
equal to the streaming oracle. The Triton kernel is a line-for-line transcription
of that mirror, so the algorithm is validated on CPU; only Triton syntax/launch
is left for the GPU-gated test.

### Data flow

```
decode query  ─┐
               │   fp store  ──────────────────────────►  fp tile(s)  ─┐
WindowedCache  │                                                        ├─► _OnlineSoftmax
.decode_       ├── QuantizedStore.iter_active_windows() ──► one window ─┤    (fp32 running
attention()    │      (int4 codes → dequant → RoPE)         at a time  ─┘     max/denom/acc)
               │                                                              │
               └──────────────────────────────────────────────►  attn output ┘  [1, H_q, 1, D]
```

The full `[H_kv, T_total, D]` effective K/V is **never built**.

---

## 4. Step-by-step: how to run it

Everything below runs on **CPU**, needs no model download, and no GPU. The only
`transformers` dependency is its RoPE helper, which the project already requires.

### 4a. Run the demo

```bash
# from the repo root (C:\StickyKV)
python scripts/demo_gemv_tiling.py
```

Expected: a correctness line (`max |tiled - materialize|` ~1e-7 in fp32) reading
`PASS`, and a peak-resident line reading `1` window.

Stress it with a large Q tier and GQA:

```bash
python scripts/demo_gemv_tiling.py --q-windows 64 --window 32 --heads 8 --rep 4
python scripts/demo_gemv_tiling.py --dtype fp16 --q-windows 32
```

Flags: `--fp-windows --q-windows --window --sink --heads --rep --dim --dtype {fp32,fp16} --seed`.

### 4b. Run the tests

```bash
# the reference kernel suite
python -m pytest tests/test_gemv.py -q

# the Triton-path suite: CPU-validatable parts (mirror==oracle, marshalling,
# dispatcher) run here; the 4 GPU kernel tests self-skip without CUDA+triton.
python -m pytest tests/test_gemv_triton.py -q

# the quant suites it builds on (no regressions)
python -m pytest tests/test_quant.py tests/test_quant_cache.py -q

# the whole CPU suite
python -m pytest -q -m "not gpu"
```

Expected: `tests/test_gemv.py` → **22 passed**; `tests/test_gemv_triton.py` →
**16 passed, 4 skipped**; full CPU suite → **303 passed, 4 deselected**.

**Validate the Triton kernel on a GPU box** (CUDA + `pip install triton`):

```bash
python -m pytest -m gpu tests/test_gemv_triton.py -q
```

This runs `test_triton_kernel_matches_oracle` (fp32/fp16 × GQA 1/4), asserting the
fused kernel equals the streaming oracle to fp tolerance.

### 4c. Call it from code

```python
# query_states: [1, H_q, 1, D], the post-RoPE decode query for one step.
attn = cache.decode_attention(layer_idx, query_states, scaling=head_dim ** -0.5)
# attn: [1, H_q, 1, D] — equals materialize→softmax to fp32 tolerance, but the
# int4 Q tier is consumed one window at a time (no full-tier fp16 blow-up).
```

`scaling` defaults to `D ** -0.5` if omitted. At `quant_ratio == 0` (or an empty
Q tier) `decode_attention` reduces to plain fp attention over the fp store, so it
is safe to call unconditionally on the decode path.

Or call the kernel directly (no cache):

```python
from modules.quant import tiled_gemv_attention
out = tiled_gemv_attention(query, fp_keys, fp_values, store, rope_module, scaling)
#   query: [H_q, 1, D] or [1, H_q, 1, D]; fp_keys/fp_values: [H_kv, T_fp, D];
#   store: a QuantizedStore (or None); returns the matching shape.
```

---

## 5. Scope and limits

- **Decode only** (`T_q == 1`). Prefill stays on the materialize path by design
  (`design.md` §11) — the Q tier is a decode-phase construct. Passing a
  multi-token query raises `NotImplementedError`.
- **Batch size 1** (v1; `design.md` §10). B > 1 raises.
- **GQA** is handled without a `repeat_kv` copy: query heads are viewed as
  `(H_kv, n_rep)` and the tile MAC broadcasts over `n_rep` — the same trick the
  flash score hook uses.
- **fp32 accumulation** regardless of KV dtype; output is cast back to the query
  dtype. Tolerances: fp32 KV ~1e-5 vs the anchor; fp16 KV ~1e-3 (fp16
  dequant/RoPE rounding).

---

## 6. Integration status (what's validated, what's gated)

Two independent gates, both consistent with the project's CPU-now / GPU-gated
posture (the dev box is CPU-only, transformers 5.8.1; see `design.md`
"Environment caveat"):

**A. The Triton kernel's on-GPU execution.** Authored and CPU-validated *at the
algorithm level* (mirror == oracle), but **not yet run on a GPU**. Close it with
`pytest -m gpu tests/test_gemv_triton.py` on a CUDA box with triton installed.
Until then `gemv_decode` uses the exact PyTorch reference — so every path stays
correct, just without the kernel's HBM-traffic win.

**B. Live `model.generate(...)` routing.** `decode_attention` is the **tested**
integration surface and matches the materialize path bit-for-bit-close through
the real `WindowedCache` (`test_cache_decode_attention_matches_materialize`, both
backends). Making a real generate call actually route its decode attention
through it additionally requires registering a custom attention function (HF
`ALL_ATTENTION_FUNCTIONS`) so the output replaces HF's SDPA/flash output, and
validating end-to-end on the **transformers 4.47.1** target env — the same gate
as Suite A/C and LongBench at int4.

---

## 7. Relationship to the kernel roadmap

- **Phase 1** (shipped): materialize-then-interleave; dequant the whole Q tier,
  glue, attend. Simple, CPU-testable, the correctness anchor.
- **Phase 2** (this): fused dequant-inside-attention. Shipped as a portable
  PyTorch reference **and** the fused Triton decode kernel
  ([`gemv_triton.py`](modules/quant/gemv_triton.py)) — window = tile = scale
  group; loads int4 codes tile-by-tile, dequantizes + RoPEs **in registers**, and
  MACs, so only int4 codes ever touch HBM. Kernel authored + CPU-validated at the
  algorithm level; GPU execution pending the `pytest -m gpu` run.
- **Phase 3** (future): FlashInfer / Atom-style paged int4 decode. Deferred until
  Phase 2 is profiled.
