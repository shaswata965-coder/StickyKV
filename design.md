# StickyKV — Quantization Design

Two-tier windowed KV cache: **top-K** windows in full precision (fp16) plus
**top-Q** windows in **int4** (hand-rolled KIVI-style), with a **per-window pinned
scale/zero-point**. The int4 (Q) tier stores keys **pre-RoPE**; RoPE is applied
fresh at read using each window's current contiguous positions. This document is the
implementation spec. The decision record — original prompt, amendment log, rejected
alternatives, and rationale — lives in [design_history.md](design_history.md).

---

## 1. Architecture overview

A window is a fixed-size chunk of the sequence (`window_size` tokens). Every
`window_size` steps the cache scores each window by accumulated attention and ranks
them into three outcomes:

- **K tier (fp16)** — the highest-ranked windows, plus the always-kept **sink**
  (first tokens) and **local** (most recent) windows. Stored full precision.
- **Q tier (int4)** — the next band of windows: not good enough for fp16 but too
  useful to drop. Stored quantized at ¼ the memory.
- **Dropped** — everything else.

The two tiers live in two separate **gap-free** dense stores (§4). Windows migrate
between tiers by ranking (promotion K←Q, demotion K→Q) and can be dropped. The cache
compacts and re-rotates survivors on **every** eviction, jointly across both tiers
(§5). Both attention backends (`windowed_cache`, `windowed_eager_cache`) share this
logic (§9).

The design is realised in three kernel phases (§11); Phase 1 (materialize-then-
concat) is the shippable v1 and is fully CPU-testable.

---

## 2. Quantization scheme

- **Granularity.** Keys are quantized **per-channel at the window-index level** —
  one scale/zero per `(head, channel, window)`. Values are quantized **per-token**.
  Quant error is set by a group's **dynamic range (max−min), not its count**: a
  single global scale is pinned by the largest outlier and obliterates small/median
  values, so groups are kept fine to localise range. But not arbitrarily fine — each
  group costs a scale + zero, so over-fine grouping eats the int4 savings.
- **Affine, asymmetric.** The distributions are skewed, so use asymmetric affine
  quantization. Validate **int8 first**, then int4.
- **Pinned grid.** Each window's **scale and zero-point are pinned at first
  quantization and reused until eviction**. The affine grid is therefore fixed, so
  `quant(dequant(c)) = c` exactly — zero drift, no compounding across re-reads (§3,
  §8).
- **Packing.** Start with **unpacked int8 codes** (so `torch.gather` works token-wise
  and the module is simple to validate); switch to **nibble-packed int4 with whole-
  window block selection** once correctness is established.
- **Outliers.** Fine grouping cannot kill **intra-group** outliers. At int4, optional
  outlier handling (dense-and-sparse retention of the top ~1% channels in fp, or a
  Hadamard rotation to spread outliers) may be needed. The specific strategy is an
  **open question** (§12), not required for the int8 milestone.

The quantizer is a shared module (a hand-rolled KIVI-style quantizer); the exact
affine formula and int4 packing layout must be pinned against a KIVI reference before
coding (§12).

---

## 3. No re-quantization; scoring is read-path only

Past KV is immutable. There is **no re-quantization** of stored windows:

- The Q tier is **dequantized for read/attention each step** so it continues to
  accrue `window_scores`. This dequant-for-scoring is a **read-path cost** (§8), not
  a re-quant.
- The new/local window is born in fp16 and is quantized **at most once** — only if it
  is later demoted into the Q tier.
- **Promotion/demotion decisions are pure score-ranking arithmetic** ("where does
  this window land in the ranking?"). No dequantization and no concatenation is
  needed for the *decision*; those happen only in the attention read path.

---

## 4. Two dense stores (not a zero-padded tensor)

Two separate, gap-free dense stores per layer:

- **fp store** — `[B, H_kv, T_fp, D]` fp16 keys/values + `position_ids`. Keys are
  rotated in place.
- **Q store** — int4 codes + per-window scales/zeros + `position_ids`. Keys are
  stored **pre-RoPE** and rotated **at read** using each window's current contiguous
  position.

A full-length fp tensor with zeros in the Q slots is **rejected**: it wastes memory
and zero keys are not softmax-neutral (`exp(q·0) = 1`). RoPE needs only a
`position_id`, not physical co-location, so at read the dequantized Q keys can simply
**concatenate `[fp ‖ Q]`** in any order (§5, "order-free"). The shared cross-store
layout is a **logical index/tier map** (the per-window ledger, §6), not a tensor. A
window's tier is implicit: tier *is* which store holds it.

---

## 5. Eviction cycle and the interleaved position map

Every eviction runs a single compaction that spans **both tiers jointly**. Fp-only
compaction is incorrect: if fp windows W1, W5 survive with Q window W2 chronologically
between them, appending W2 after W5 in position space tells the model W2 is the most
recent context, corrupting every Q·Kᵀ dot product involving W2.

**The cycle, per eviction:**

1. **Rank** all windows by accumulated `window_scores`.
2. **Assign tiers.** Top `N_fp` windows (including sink + local) → fp; next `N_q` →
   Q; the rest → dropped (`N_fp`, `N_q` from the budget resolver, §7).
3. **Move boundary-crossers.**
   - **Demote (K→Q):** un-rotate the window's fp keys **once** (reuse the
     `rerotate_keys` un-rotate half), quantize against a freshly-pinned grid, append
     to the Q store, remove from the fp store. (A window that has been demoted before
     re-uses its pinned grid by identity — §8.)
   - **Promote (Q→K):** dequantize the window's codes, append to the fp store, remove
     from the Q store.
4. **Build the interleaved position map.** Merge **all** surviving windows (both
   tiers) sorted by `original_window_id`; assign one contiguous position map
   `arange(T_total)` across the merged set, where `T_total = T_fp + T_q`. Fp windows
   take their slots in this map; Q windows take theirs. Fp slots therefore **skip
   over** the positions occupied by interleaved Q windows.
5. **Re-rotate the fp tier** to its slots in the interleaved map — which may be
   non-contiguous (gaps where Q windows sit). This uses `rerotate_keys` with the
   **explicit interleaved target positions** for each fp survivor.
6. **Update the Q ledger.** Write each surviving Q window's new `position_range` from
   the map. The codes and scale/zero **do not change** — only the integer
   `position_range` is updated. Cost: O(Q_windows) integer assignments, no tensor
   movement, no re-quant.
7. **Override the query position** to `T_total` (not `T_fp`) via
   `utils.position_override`, so query↔key relative distance is exact across both
   tiers. New tokens append at the overridden compacted position.

**Why pre-RoPE.** Because positions are rebased on *every* eviction, baking RoPE into
the Q codes would force un-rotate → re-rotate → re-quant each cycle; re-rotation
changes the values, so re-quantizing against the pinned grid would no longer be
idempotent and error would accumulate. Storing codes **un-rotated** and stamping RoPE
only at read keeps the codes frozen forever: pinned-grid idempotence holds and there
is zero RoPE-driven quant-error accumulation. Values carry no RoPE in either tier
(asymmetric store).

**Order-free concat.** The read-time concat `[fp_store ‖ dequant_Q]` may appear in any
physical order: RoPE has already baked each key's logical position into its values, so
Q·Kᵀ is correct regardless of where a key physically sits.

### New/changed primitives (relative to the current cache)

- `build_interleaved_position_map(fp_window_ids, q_window_ids, window_size, num_sink)`
  — sorts all surviving window ids jointly, assigns contiguous positions, returns
  **(a)** the fp-survivor target-position tensor for `rerotate_keys` and **(b)** the
  per-Q-window `position_range` assignments for the ledger. Must operate **per row**
  (rows may evict different windows, as the current cache already does).
- `rerotate_keys(rope, old_pos, new_pos)` — gains an explicit `new_pos` argument (was
  implicit `arange(T_fp)`). **Note:** its trailing `position_ids` bookkeeping must be
  set to the interleaved (possibly gappy) fp positions, **not** `arange(T_retained)`,
  or the next eviction snapshots wrong "old" angles.
- `position_override.py` — `cache_position` uses `T_total` (fp + Q tokens), not
  `T_fp`.

---

## 6. Per-window ledger

A small record keyed by `original_window_id` tracks each surviving Q window across
evictions. The fp tier needs no ledger — it is a plain dense tensor.

| field | mutable? | purpose |
|---|---|---|
| `original_window_id` | no | chronological identity; used for the interleaved sort |
| `codes` (int4) | no | packed quantized bits; never change after demotion |
| `scale`, `zero` | no | pinned affine grid; set once at demotion |
| `offset` | yes | byte offset into the Q store; shifts as the Q store compacts |
| `position_range` | yes | current contiguous positions in the interleaved map; updated every eviction via `build_interleaved_position_map` |

`position_range` is the one field that changes at eviction cadence, and it is what
the read path feeds to RoPE. The ledger update at eviction is O(Q_windows) integer
assignments — no tensor movement, no re-quant.

---

## 7. Tier-aware budget resolver

A `quant_ratio` knob `q` splits the **memory** budget (not the window count) between
tiers. Let `M_full` be the full-cache memory and `β` (the existing `cache_budget`)
the retained fraction:

```
M_budget = β · M_full
M_fp     = (1 − q) · M_budget
M_q      = q · M_budget

N_fp = M_fp / b_fp          # fp windows
N_q  = M_q  / b_q           # int4 windows
```

where `b_fp` is fp bytes-per-window and

```
b_q ≈ ¼ · b_fp  +  per-window key-scale/zero  +  per-token value-scale overhead
```

The resolver **must use `b_q`, not `b_fp`, for the Q tier** — the int4 tier holds
~4× the windows of equal fp memory (minus overhead). The scale/zero overhead term
depends on the chosen **scale dtype** (fp16 / fp8 / int8), which is an empirical knob
(§12); the resolver takes it as an input so `N_q` is deterministic for a given config.

**Sink + local windows stay fp**, inside `M_fp`: `top-K-fp = N_fp − (sink + local)`.

*Example* (β = 0.25, q = 0.5): 12.5% fp + 12.5% int4 ⇒ `N_q ≈ 4·N_fp` ⇒ ~62.5% of
windows retained at 25% of full memory.

This extends the existing byte-based `resolve()` (which already computes
`bytes_per_token`, `total_budget_bytes`, `top_k_windows`). New config knobs: `β`
(exists as `cache_budget`), `q`, bit-width, group size, scale dtype.

---

## 8. Read / attention path and per-step cost

**Read path (v1, materialize).** For each Q window: look it up in the ledger, take
the int4 codes, dequantize to fp16, apply RoPE at the window's current
`position_range`, then concat `[fp ‖ dequant-Q]` and hand the result to the standard
attention path. The fp tier is already rotated and ready. `update()` returns one
normal fp tensor.

**Cost.** Recent/local + sink + top-K stay fp, so the most-attended tokens skip the
slow path. The per-step Q cost is: dequant + one RoPE apply (arithmetic on already-
dequantized data — bandwidth-trivial). The real v1 cost is the **fp16 write-back**:
the Q tier blooms to fp16 transiently, but attention runs **layer-by-layer**, so only
one layer's Q tier is live in fp16 at any moment — peak impact ≈ `(Q-fp size) /
num_layers` (~1–2% of the full cache at 32 layers), freed immediately after each
layer. Phase 2 eliminates the write-back entirely (§11).

**Scoring.** Because `update()` returns the concatenated effective K/V, attention (and
the eager scoring hook) attends over the dequantized Q keys and both tiers accrue
`window_scores` with no change to the scorer. The scoring path must map the
per-window attention weights back to **both** stores' windows (the concat is order-
free, so the mapping is by window identity, not physical position).

**Benchmark gate.** Suite C (`perf_runner.py`) must confirm memory savings outweigh
TPOT impact in v1 before moving to Phase 2.

---

## 9. Backend mirroring

"Mirror" means the **byte-identical** `cache.py` / `state.py` (+ the new shared quant
module and ledger), **not** `hooks.py`. The two backends diverge only in hooks: flash
recomputes scores via an auxiliary SDPA; eager reads materialized attention weights —
no aux SDPA is added to eager.

- The query-position override is already shared (`utils.position_override`, installed
  from both backends' `install_score_hooks`).
- Shared `update()` returns the effective K/V `[fp ‖ dequant+rotate Q]` (Q tier is
  pre-RoPE, so the read path dequantizes then applies RoPE at each window's current
  positions), so **eager scoring needs no change**.
- The flash aux SDPA sources the same effective K via a shared
  `materialize_effective_kv` helper.
- Transient dequant is **per-layer** (§8).

**Implementation note:** `update()` currently returns the live `state.key_states /
value_states`. With the Q tier it returns a freshly-built concat instead; callers must
not assume the returned tensor aliases the stored fp cache.

---

## 10. Key design choices

- **Full bidirectional promotion in v1** (per the original prompt). This accepts the
  ledger bookkeeping and the score-feedback risk; both are instrumented via promotion-
  frequency telemetry and **Suite A Jaccard-vs-fp-only** over long sequences. (The
  score-feedback loop and the not-chosen fallback are documented in
  [design_history.md](design_history.md).)
- **Pinned grid by identity kills oscillation.** A window retains its pinned grid **by
  identity, even through a promotion** → a promote→demote round trip re-quantizes
  against the old grid → idempotent → identical codes → zero added error. Explicit
  hysteresis is optional and deferred (§12).
- **Quant group = the eviction window.** This pins one grid per window, which
  promotion requires. `window_size`, bit-width, and scale dtype are empirical knobs
  swept in Suite C / LongBench — **no hardcoded floor**; pick by measured effective-
  bits-vs-quality. Effective key bits ≈ `4 + 32/window_size` (expectation-setting,
  not a rule).

---

## 11. Kernel roadmap (three phases)

### Phase 1 — v1: materialize-then-concat (ship first)

Dequantize the entire Q store to fp16, concat with the fp store `[fp ‖ dequant-Q]`,
pass to the standard attention path unchanged. No custom kernels; fully CPU-testable;
correctness is the only goal.

- **Q-tier RoPE:** pre-RoPE — codes stored un-rotated; read path is `dequantize →
  apply RoPE at the window's current contiguous positions → concat`.
- **Memory peak:** a transient fp16 copy of one layer's Q tier at a time
  (≈ `(Q-fp size)/num_layers`), freed immediately. The fp16 write-back is the real
  cost (~4× the int4 read); Phase 2 eliminates it.
- **Exit criterion:** Suite C confirms net memory savings; Suite A Jaccard holds;
  LongBench quality acceptable at int8, then int4. Only then move to Phase 2.

### Phase 2 — Triton GEMV tile: fused dequant-inside-attention (future work)

A Triton decode kernel that loads int4 codes tile-by-tile, dequantizes to fp16 **in
registers**, and runs `Q·Kᵀ` before any write-back. The fp16 materialization is
eliminated — only int4 codes are read from HBM.

- **Storage: pre-RoPE** (same as Phase 1). Tile kernel: `load int4 → unpack → scale →
  apply RoPE from cos/sin → MAC`. RoPE is arithmetic on already-loaded data — zero
  extra memory traffic in the bandwidth-bound decode regime. `cos/sin` are recomputed
  for a window's current positions whenever an eviction rebases them (cheap — the
  codes never change).
- **Scope:** decode path only (GEMV, one query token at a time). Prefill stays on the
  Phase 1 materialize path — fine, since the Q tier is a decode-phase construct.
- **Layout fit:** tile boundary = window boundary = scale-group boundary. One tile
  reads one window's codes and one pinned `(scale, zero)` — no cross-tile scale
  bookkeeping.

### Phase 3 — FlashInfer integration (production ceiling, not in scope)

Replace the custom GEMV tile with FlashInfer's paged quantized decode attention
(online softmax, GQA, paged blocks, int4/fp8 native). Requires aligning
`QuantizedStore`'s block layout with FlashInfer's paged KV convention. Strictly
better than Phase 2 but adds a significant dependency and layout constraint. Deferred
until Phase 2 is profiled.

---

## 12. Open questions (resolve before the affected milestone)

- **Outlier strategy (blocks int4, not int8).** Dense-and-sparse (top ~1% channels in
  fp, KVQuant) vs Hadamard rotation (RotateKV/QuaRot) vs none. Decide by measured int4
  quality on LongBench.
- **Scale dtype (blocks a deterministic `b_q`).** fp16 / fp8 / int8 for the pinned
  scale/zero. Swept in Suite C; the resolver needs a default to compute `N_q`.
- **Quantizer numerics.** Pin the exact asymmetric affine quant/dequant formula and
  the int4 nibble-packing layout against a KIVI reference before coding the module.
- **B>1 composition.** The v1 read path relies on `position_override`, which is a
  **B=1 construct** (see its docstring), and `build_interleaved_position_map` must run
  per-row. v1 quant is therefore **B=1-only** unless it is explicitly integrated with
  the ragged left-padded batching design. State the intended scope before coding.
- **Explicit hysteresis.** Deferred; revisit only if Suite A Jaccard shows measurable
  K/Q boundary churn.

---

## Environment caveat

The **target** across eval devices is **transformers 4.47.1**; `environment.yml` is
pinned to the 4.47.x line. transformers 5.x builds the causal mask via
`create_causal_mask` → `Cache.get_mask_sizes()`, which `WindowedCache` does not
implement, so a full-model forward crashes on 5.x. The current dev machine runs
transformers 5.8.1 / torch 2.12 / Python 3.12, so until it is brought to 4.47.x,
verify cache/quant logic via **CPU unit tests** (`pytest -m "not gpu"`), not full-
model runs. (`utils/cache_factory.py` refuses to run on > 4.47.1 rather than crash
mid-run.)

---

## Implementation outline

New `QuantizedStore` + hand-rolled KIVI-style quantizer module; two-tier
`update()` / eviction with the per-window ledger (§5, §6); `build_interleaved_
position_map` + `rerotate_keys(new_pos)` change; `materialize_effective_kv` helper;
tier-aware budget resolver (§7); mirrored into both backends (§9).

CPU unit tests: round-trip error, pinned-grid idempotence, position-invariance,
interleaved-map correctness (fp gaps over Q slots), flash/eager parity.

Gates: Suite C (peak memory + throughput/TPOT), Suite A (Jaccard drift vs fp-only),
LongBench (quality at int8, then int4).

---

## Appendix: the whole thing in plain English

**The problem.** When the model reads a long prompt and generates, it remembers every
token it has seen — that memory is the KV cache. It keeps growing and eventually eats
the whole GPU, so we have to throw stuff away. The whole game is throwing away the
right stuff and keeping what matters.

**Windows.** We chop the sequence into fixed-size chunks called windows (say 32
tokens each). Every `window_size` steps we pause, look at how much attention each
window has been pulling, and rank them. A couple of windows never get ranked — the
sink (first few tokens) and the local window (the most recent one) are always kept.

**Three buckets instead of two.** Normally a window is either kept or deleted. We add
a middle bucket. The best windows stay in full precision fp16 — that's the K tier.
The ones not good enough for fp16 but still too useful to throw away, we squeeze down
to int4, a quarter of the memory — that's the Q tier. Everything else is dropped.

**What happens at every eviction.**

1. We rank all the windows.
2. The windows crossing into the Q tier get quantized — but right before we quantize,
   we strip RoPE off them (RoPE is the position stamp on a token). We store the
   stripped, un-stamped version.
3. The survivors get squished together so there are no gaps, and we renumber their
   positions from zero.
4. The renumbering covers **both tiers at once**. Say windows 1 and 5 stay in fp16 and
   window 3 sits between them in the Q tier. We can't renumber 1 and 5 and tack 3 on
   at the end — that would tell the model window 3 is the newest thing it saw, which
   is false. So we sort all survivors back into original order, lay out one shared set
   of positions across both tiers, and the fp windows leave a gap in position space
   where the Q windows live.
5. Every eviction we update the `position_range` of every Q window in the ledger. The
   codes never change; the scale never changes; only this one position number changes.
   It is one cheap integer write per window — and it must never be skipped, or the
   window gets stamped to the wrong place at read time.

**Why un-stamped (pre-RoPE).** Because we renumber on *every* eviction. If we baked
the stamp into the codes, every cycle we'd have to un-stamp, re-stamp at the new
position, and re-quantize — and re-quantizing piles on a little error each time until
the window turns to mush. Storing codes un-stamped and stamping fresh only at read
means the codes are frozen forever and never drift.

**The forward pass.**

- The fp16 windows are already stamped and ready.
- For each Q window we look it up in the ledger, grab the int4 codes, blow them back
  up to fp16 (dequantize), and stamp them with RoPE using the `position_range` we've
  kept fresh.
- We glue the fp16 and freshly-stamped Q windows into one tensor and hand it to normal
  attention. The glue order doesn't matter — each key already carries its correct
  position inside its own values, so attention gets the right distances no matter
  where a key physically sits.

In Phase 1 we do this the simple way — blow up the whole Q tier, glue, attend. In
Phase 2 we do the blow-up one window at a time *inside* the attention kernel, so the
fp16 version is never written to memory; only the small int4 version ever lives in
HBM.
