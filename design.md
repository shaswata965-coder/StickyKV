# StickyKV — Quantization Design

## Initial prompt (verbatim)

> To integrate quantization into our wqorkflow what would be the major challenges?
> How I intend to integrate quantization:
> we will be retaining top K+ top Q windows, these q windows will be stored in quantized form
> While decoding when we have a new window it will be compared against full precision and then dequantized window, if any dequantized window is more important than it will promoted and stored along with the full precision windows and a new window will be quantized
> During generation we will dequantize and quantize on the fly to store their Updated cummulated attention as well
> Finally we want to implement a pre Rope quantization, meaning during presses stripping of rope and re applying rope we want to do the quantization operation (if possible) so that rope does not accumulte quantization error

---

## Locked design (resolutions)

Two-tier windowed KV cache: **top-K** windows in full precision + **top-Q** windows
in **int4** (hand-rolled KIVI-style), with **per-window pinned scale/zero**.
Q-tier keys are stored **pre-RoPE** in **both** phases (see AMENDMENT 2): the cache
re-rotates survivors on every eviction, so pre-RoPE is required to keep the pinned
grid idempotent (post-RoPE would accumulate quant error), and it also gives better
int4 quality. Resolutions below are the agreed design; full challenge analysis lives
in the plan file (`~/.claude/plans/to-integrate-quantization-into-witty-stonebraker.md`).

> ## ⚠️ AMENDMENT 2 — rerotation RESTORED; compact + re-rotate + query-position override (supersedes the `f80326b` amendment)
>
> Verified against the real NVIDIA/kvpress source: KVPress **does** re-rotate
> evicted-survivor keys (`KeyRerotationPress` rebases survivors to contiguous
> `[0..N_survivor-1]`, computes `delta_pos = idx − selected_positions`, and
> rebuilds cos/sin) **and** overrides the query position in its pipeline
> (`context_length = cache.get_seq_length()` ⇒ the query lands at the compacted
> length, the "N_survivor+1" slot — not its original position). The earlier
> `f80326b` amendment (keep-original-positions, rerotation off) was based on the
> **opposite, incorrect** reading and has been **reverted**. On **every** eviction
> the cache now: (1) compacts survivors contiguous, (2) **re-rotates** their keys
> to contiguous positions (`state.rerotate_keys`), and (3) a forward pre-hook
> (`utils.position_override`) overrides the query's `position_ids`/`cache_position`
> to the compacted cache length each step. The `rerotate_on_evict` knob is gone —
> this is the only path. Because the query position is set **explicitly**, this is
> correct independent of the transformers version. Implications for quantization:
>
> - **The original-prompt goal "pre-RoPE so RoPE does not accumulate quantization
>   error" (#4) is back in force — and is now a *correctness* requirement for the
>   Q tier, not merely a quality lever.** A strip→re-rotate cycle runs at **every**
>   eviction. If Q-tier keys were stored **post-RoPE**, keeping them consistent
>   with the rebased positions would require
>   dequant→un-rotate→re-rotate→re-quant each eviction; re-rotation changes the
>   values, so re-quantizing against the pinned grid is **no longer idempotent**
>   and quantization error **accumulates** across evictions (breaking #13).
>   Therefore the Q tier stores **pre-RoPE** codes: un-rotated **once** at
>   demotion, pinned grid, with RoPE applied **fresh at read** using the window's
>   current (contiguous) positions. The stored codes never change across evictions
>   — only the cos/sin applied at read do — so pinned-grid idempotence (#13) holds
>   and there is **zero** rope-driven quant-error accumulation.
> - **#2 / #8 — pre-RoPE is now the Q-tier default in BOTH phases** (was: post-RoPE
>   v1, pre-RoPE Phase 2). v1 materialize path: `dequant → apply RoPE at current
>   positions → concat [fp ‖ Q]` (one extra RoPE apply on already-dequantized data
>   — no custom kernel). Phase 2 tile: `load int4 → unpack → scale → apply RoPE
>   from cos/sin → MAC`. The demotion-time un-rotate **reuses `rerotate_keys`'
>   un-rotate half**, which now always runs.
> - **#3 / #9 — positions ARE rebased to contiguous every eviction** (not "never
>   rebased"). The Q store records each surviving window's **current** position
>   range and updates it at each eviction; because the codes are pre-RoPE
>   (position-independent) this costs only a cos/sin recompute at read, never a
>   re-quant. The query carries the **overridden compacted** `cache_position`, so
>   query↔key relative phase is exact (not the old compaction approximation).
> - **#7 / #10 — per-step Q cost gains one RoPE apply** (pre-RoPE read), still
>   bandwidth-trivial; the fp16 write-back remains the real v1 cost Phase 2
>   eliminates.
> - **G5 restored:** both tiers re-rotate every eviction. The fp tier re-rotates
>   in place (negligible fp error); the Q tier avoids compounding precisely
>   *because* it stores pre-RoPE codes (above).
> - **Both backends mirror this** (`windowed_cache` + `windowed_eager_cache`),
>   consistent with #10.
>
> The affected resolutions below (#2, #3, #7, #8, #9, #10) have been updated inline
> to match this amendment; no conflicting guidance remains.

1. **Quantization granularity.** Keys quantized **per-channel at the window-index
   level** (one scale/zero per `(head, channel, window)`); values **per-token**.
2. **No compounding from re-quant.** Each window's **scale + zero-point are pinned
   at first quantization and reused until eviction** → fixed affine grid →
   `quant(dequant(c)) = c` exactly (zero drift).
3. **#4 — No re-quantization needed.** Past KV is immutable. Dequantize the Q tier
   **for read/attention each step** so it accrues `window_scores`; the new/local
   window is born in fp and quantized **at most once** (only if later demoted).
   Dequant-for-scoring is a read-path cost (see #7), not a re-quant.
4. **#5 — Comparison is just ranking.** Promotion/demotion is pure score-ranking
   arithmetic ("where does the new window land?"). **No dequantization and no
   concatenation for the decision** — those happen only in the attention read path.
5. **#6 — Granularity, outliers, overhead.** Quant error is set by a group's
   **dynamic range (max−min), not its count**; one global scale is pinned by the
   largest outlier and obliterates small/median values. So group finely (keys
   per-channel per-window, values per-token) to localize range — but not
   arbitrarily: each group costs a scale+zero, so too-fine groups eat the int4
   savings. Fine grouping still can't kill **intra-group** outliers; at int4 add
   optional **dense-and-sparse** outlier retention (top ~1% channels in fp, KVQuant)
   or a **Hadamard rotation** to spread them (RotateKV/QuaRot). Outlier strategy
   TBD — to be decided separately. Use **asymmetric** quant for the skewed
   distributions; validate **int8 first**, then int4.
6. **#11 — Tier-aware budget via a quant ratio.** Add a `quant_ratio` knob `q`
   that splits the **memory** budget (not the window count) between tiers:
   `M_budget = β·M_full`, `M_fp = (1−q)·M_budget`, `M_q = q·M_budget`. Convert each
   tier's memory to windows with **its own** bytes-per-window:
   `N_fp = M_fp / b_fp`, `N_q = M_q / b_q` where
   `b_q ≈ ¼·b_fp + per-window key-scale/zero + per-token value-scale overhead`.
   The scale/zero overhead term depends on the chosen **scale dtype** (fp16, fp8,
   or int8) — treated as an **empirical knob swept in Suite C**, not a constant.
   The int4 tier holds ~4× the windows of equal fp memory (minus overhead) — the
   resolver **must use `b_q`, not `b_fp`, for the Q tier**. Example (β=0.25, q=0.5):
   12.5% fp + 12.5% int4 ⇒ N_q≈4·N_fp ⇒ ~62.5% of windows at 25% memory. **Sink +
   local windows stay fp inside `M_fp`** (`top-K-fp = N_fp − (sink + local)`). Expose
   `β`, `q`, bit-width, group size as config knobs.
7. **#1 — Two dense stores, not a zero-padded tensor.** An **fp store**
   (`[B,H_kv,T_fp,D]` + `position_ids`) and a separate **Q store** (int4 codes +
   scales/zeros + `position_ids`), both **gap-free**. *Rejected:* a full-length fp
   tensor with zeros in Q slots — wastes memory and zero keys aren't softmax-neutral
   (`exp(q·0)=1`). RoPE needs only `position_id`, not co-location: the fp tier is
   rotated in place, the Q tier stores **pre-RoPE** codes and is rotated **at read**
   using each window's current contiguous position, so the dequantized result can
   simply **concatenate `[fp ‖ Q]`** (order-free during decode). Shared layout is a
   **logical index/tier map** (the per-window record, #9), not a tensor.
8. **#2 — Q-tier RoPE strategy: pre-RoPE in both phases (per AMENDMENT 2).**
   A strip→re-rotate cycle runs at **every** eviction, so the Q tier stores
   **pre-RoPE** codes: un-rotate **once** at demotion (reuse `rerotate_keys`'
   un-rotate half, which now always runs), pin the grid, and apply RoPE **fresh at
   read** using the window's current contiguous positions. This keeps the pinned
   codes idempotent across evictions; post-RoPE storage would instead accumulate
   quant error through repeated dequant→re-rotate→re-quant (see AMENDMENT 2).
   **v1 (materialize):** `dequant → apply RoPE at current positions → concat
   [fp ‖ Q]` — no custom kernel. **Phase 2 (Triton tile):** `load int4 → unpack →
   scale → apply RoPE from cos/sin → MAC` — RoPE is arithmetic on already-loaded
   data, **zero extra memory traffic** in the bandwidth-bound regime. Pre-RoPE also
   yields better int4 quality (KVQuant: consistent per-channel outliers before
   rotation, smeared after). `update()` returns one normal fp tensor in both
   phases; values carry no RoPE (asymmetric store).
9. **#3 — Positions rebased to contiguous every eviction (per AMENDMENT 2).**
   Each eviction re-rotates survivors to contiguous `arange(T_retained)` and the
   query's `cache_position` is **overridden** to the compacted length, so
   query↔key relative distances are **exact** (not the old compaction
   approximation). The Q store records each surviving window's **current**
   (contiguous) position range and refreshes it at each eviction; because the
   codes are **pre-RoPE (position-independent)**, rebasing costs only a cos/sin
   recompute at read — never a re-quant — so the pinned grid survives rebasing.
   New tokens append at the (overridden) compacted position.
10. **#7 — Per-step Q-tier cost: accepted, mitigated, measured.** Recent/local + sink
    + top-K stay fp, so the most-attended tokens skip the slow path. **v1 (pre-RoPE,
    materialize path): dequant + one RoPE apply per step** — the Q tier blooms to
    fp16 transiently per layer (modest: `(Q-fp size)/num_layers`), then freed; the
    RoPE apply is arithmetic on the already-dequantized tensor (bandwidth-trivial).
    **Phase 2 (Triton tile): eliminates the fp16 write-back entirely**; RoPE moves
    in-tile (still zero extra memory traffic). Benchmark gate: Suite C
    (`perf_runner.py`) must confirm memory savings outweigh TPOT impact in v1 before
    moving to Phase 2. (SKVQ-style recent-window-in-fp.)
11. **#8 — Gather is a non-issue; grids are stable.** Start with **unpacked int8
    codes** → `torch.gather` works token-wise; when packing to nibbles later, switch
    the Q store to **whole-window block selection**. Since positions are **never
    rebased** (AMENDMENT), a window's codes + scale/zero stay valid across evictions
    unconditionally — nothing about a window changes at compaction except that it may
    be dropped. Key the grid to window identity.
12. **#9 — Tier flag is implicit; the per-window record is lightweight.** Tier *is*
    which store holds a window. A small record keyed by `original_window_id` carries
    each surviving Q window's **pinned grid `(codes, scale, zero)` across evictions**
    (#12) and supports the **current-store vs new-assignment diff** so only
    boundary-crossing windows promote/demote. It's `original_window_ids` + the Q
    store's `(offset, scale, zero)`.
13. **#12 — Pinned grid by identity kills oscillation.** Retain a window's pinned
    grid **by identity, even through a promotion** → promote→demote re-quantizes
    against the old grid → **idempotent → identical codes → zero added error**.
    Hysteresis optional. **transformers 4.47.1 target across devices**;
    `environment.yml` is already pinned to `>=4.47,<4.48`. ⚠️ *See "Environment
    caveat" below — the current dev machine actually runs 5.8.1.*
14. **#10 — Mirror the shared twins, keep hooks divergent.** "Mirror" = the
    byte-identical `cache.py`/`state.py` (+ new shared quant module), **not**
    `hooks.py` (flash recomputes via aux SDPA; eager reads materialized weights — we
    do **not** add aux SDPA to eager). The query-position override is already shared
    (`utils.position_override`, installed from both `install_score_hooks`). Shared
    `update()` returns effective K/V `[fp ‖ dequant+rotate Q]` (the Q tier is
    pre-RoPE, so the read path dequantizes then applies RoPE at the window's current
    positions), so eager scoring needs no change; the flash aux SDPA sources the same
    effective K via a shared `materialize_effective_kv` helper. Transient dequant is
    **per-layer** (modest, ~`(Q-fp size)/num_layers`).

### Locked decisions (bigger-picture review)
- **Full bidirectional promotion in v1** (per the initial prompt). Accept the
  #9/#12/#13 bookkeeping and the score-feedback risk (G4); **instrument promotion
  frequency + Suite A Jaccard-vs-fp-only over long sequences**. Documented fallback:
  one-way demotion + frozen Q-scores (not chosen).
  > **G4 explained — score-feedback loop:** `window_scores` are accumulated from
  > attention weights computed over the dequantized Q tier. Because int4 is lossy,
  > those attention weights are slightly wrong, so the score increments for Q-tier
  > windows are noisy. Windows near the K/Q score boundary are the most exposed —
  > small noise can flip a demotion decision, causing spurious promotion/demotion
  > churn. Bounded naturally because the fp tier (sink + local + top-K) dominates
  > attention mass and anchors most scores. Detected via **Suite A Jaccard**: compare
  > which windows survive in a two-tier run vs a pure-fp baseline over long sequences;
  > large divergence flags feedback drift. Hysteresis is the surgical fix if detected,
  > but is deferred (see "Rejected optimizations").
- **Quant group = the eviction window** (pins the grid per window, required by
  promotion); `window_size`, bit-width, and scale dtype (fp16/fp8/int8) are
  **empirical knobs swept in Suite C / LongBench** — **no hardcoded floor**; pick by
  measured effective-bits-vs-quality. Effective key bits ≈ `4 + 32/window_size`
  (expectation-setting, not a rule).

## Environment caveat
The dev machine currently has **transformers 5.8.1 / torch 2.12 / Python 3.12**,
which still crashes a full-model forward through `WindowedCache`
(`create_causal_mask` → `get_mask_sizes()`). The **target** across eval devices is
**4.47.1**; `environment.yml` is pinned to the 4.47.x line. Until this machine is
brought to 4.47.x, verify cache/quant logic via **CPU unit tests** (`pytest -m
"not gpu"`), not full-model runs.

## Implementation outline
New `QuantizedStore` + hand-rolled KIVI-style quantizer module; two-tier
`update()`/eviction with the per-window record; `materialize_effective_kv` helper;
tier-aware budget resolver; mirrored into both backends. CPU unit tests:
round-trip error, pinned-grid idempotence, position-invariance, flash/eager parity.
Gates: Suite C (peak memory + throughput/TPOT), Suite A (Jaccard drift), LongBench
(quality at int8 then int4).

## Considered and explicitly rejected optimizations

The following were analysed and dropped before v1. Recorded here so they are not
re-debated during implementation.

- **Async eviction** (overlap the RoPE strip + re-rotate + quantize with FFN via
  CUDA streams): a strip→re-rotate cycle now runs every eviction, but the per-step
  work is still bounded (a few newly-demoted windows). Overlapping it requires
  decoupling this step's attention from the compaction (Suite A parity break),
  raises peak memory during overlap (pre- and post-eviction buffers coexist), is
  GPU-only, and adds stream/determinism risk. **Dropped.**

- **Deferred memory movement** (flag migrations, batch copies every N steps):
  movement already happens only at eviction cadence, batched, boundary-crossers
  only. Pushing N beyond `window_size` overshoots the memory budget during deferral
  and splits logical tier from physical store (ambiguous precision in read/score
  paths). Safe substitute: hysteresis (see below). **Dropped.**

- **Prefetch next-layer Q-tier dequant during FFN** (overlap dequant with the
  adjacent layer's FFN): decode is memory-bandwidth-bound on weight loading already;
  FFN weight traffic saturates HBM, leaving no free bandwidth shadow. Running dequant
  concurrently on a second stream adds bytes moved rather than hiding them. Q-tier
  dequant is ~1% of step bandwidth — real but unhideable by scheduling. The right
  lever is eliminating the write-back (Phase 2 fused kernel), not prefetching it.
  **Dropped.**

- **Hysteresis at the K/Q boundary** (require a score margin before migrating):
  demotion costs one un-rotate + quantize and promotion one dequant — both cheap.
  With the pinned **pre-RoPE** grid, an oscillating window's codes are unchanged
  across migrations (idempotent, zero error). Hysteresis adds a tunable margin knob
  for marginal migration savings. Its only remaining value is damping the G4
  score-feedback drift, already instrumented via Suite A Jaccard. **Dropped as a v1
  concern;** revisit if Suite A shows measurable boundary churn.

## Kernel roadmap (three phases)

### Phase 1 — v1: materialize-then-concat (ship first)
Dequantize the entire Q store to fp16, concatenate with the fp store
`[fp ‖ dequant-Q]`, and pass the result to the standard attention path unchanged.
No custom kernels; fully CPU-testable; correctness is the only goal here.
- **Q-tier RoPE:** pre-RoPE — codes are stored un-rotated, so the read path is
  `dequantize → apply RoPE at the window's current contiguous positions → concat`.
  (Post-RoPE storage is unusable here: the strip→re-rotate cycle that runs every
  eviction would accumulate quant error — see AMENDMENT 2.)
- **Memory peak (the materialization concern):** yes, this creates a transient fp16
  copy of the Q tier. But attention runs **layer-by-layer**, so only one layer's Q
  tier is live in fp16 at any moment — peak impact ≈ `(Q-fp size)/num_layers`.
  At 32 layers that is ~1–2% of the full cache, not a showstopper. The transient is
  freed immediately after each layer's attention. The fp16 **write-back** is the
  real bandwidth cost (~4× the int4 read); that is what Phase 2 eliminates.
- **Exit criterion:** Suite C confirms net memory savings (steady-state int4 storage
  outweighs the per-layer transient); Suite A Jaccard holds; LongBench quality
  acceptable at int8, then int4. Only then move to Phase 2.

### Phase 2 — Triton GEMV tile: fused dequant-inside-attention (future work)
A Triton decode kernel that loads int4 codes tile-by-tile, dequantizes to fp16
**in registers**, and runs `Q·K^T` before any write-back to global memory. The
fp16 materialization is eliminated entirely — only int4 codes are read from HBM.
- **Q-tier storage: pre-RoPE** (same as Phase 1; the only viable storage under the
  rerotation methodology — post-RoPE would accumulate quant error across the
  per-eviction re-rotations, see AMENDMENT 2). Codes are un-rotated once at demotion;
  each window's current contiguous positions drive its `cos/sin`. The tile kernel is:
  `load int4 → unpack → scale → apply RoPE from cos/sin → MAC`. RoPE is pure
  arithmetic on already-loaded data — **zero extra memory traffic** in the
  bandwidth-bound decode regime; the arithmetic overhead is negligible. Also gives
  the KVQuant per-channel-outlier quality benefit. `cos/sin` are recomputed for a
  window's current positions whenever an eviction rebases them (cheap — the codes
  themselves never change).
- **Scope:** decode path only (GEMV, one query token at a time). Prefill continues
  on the Phase 1 materialize path. That is fine — the Q tier is a decode-phase
  construct (windows are demoted during generation, not prefill).
- **Layout fit:** tile boundary = window boundary = scale group boundary. One tile
  reads one window's codes and one pinned `(scale, zero)` — clean, no cross-tile
  scale bookkeeping.

### Phase 3 — FlashInfer integration (production ceiling, not in scope)
Replace the custom GEMV tile with FlashInfer's paged quantized decode attention,
which handles the full FlashAttention tile loop (online softmax, GQA, paged blocks)
with int4/fp8 natively. Requires aligning `QuantizedStore`'s block layout with
FlashInfer's paged KV convention. Strictly better than Phase 2 (handles both prefill
and decode, production-tested), but introduces a significant dependency and layout
constraint. Deferred until Phase 2 is profiled and the layout migration cost is
justified.
