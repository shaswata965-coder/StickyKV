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
in **int4** (hand-rolled KIVI-style), with **pre-RoPE** key quantization and
**per-window pinned scale/zero**. Resolutions below are the agreed design; full
challenge analysis lives in the plan file
(`~/.claude/plans/to-integrate-quantization-into-witty-stonebraker.md`).

> ## ⚠️ AMENDMENT — rerotation removed; contiguous-only, original positions (commit `f80326b`)
>
> The cache no longer re-rotates keys on eviction. `rerotate_on_evict` now defaults
> to **`False`**: `slice_and_keep` gathers survivors **contiguous in memory** but
> keeps their **ORIGINAL `position_ids`** (no rebasing to `arange`), and keys retain
> the RoPE rotation they were stored with. This matches **KVPress / H2O** and is
> correct on any transformers version (rebasing corrupts the query↔key relative
> phase since HF advances the query `cache_position` monotonically). Several
> resolutions below change as a result:
>
> - **The original-prompt goal "RoPE does not accumulate quantization error" is now
>   met by the architecture itself.** With no strip/re-apply cycle, RoPE is applied
>   **exactly once** (at prefill/gen) and never re-applied to cached keys → no
>   rope-driven error accumulation for *either* tier, quantized or not. So the
>   *accumulation* motivation for pre-RoPE is **moot**.
> - **#2 reframed — Q-tier RoPE is now a fork, defaulting to POST-RoPE.** There is no
>   free "strip gap" to quantize inside anymore. **Default (recommended): quantize the
>   already-rotated surviving keys as-is** (post-RoPE); at read **dequantize only — no
>   rotation**; demotion needs **no strip**; promotion is a plain dequant. This drops
>   the per-step rotation cost and simplifies #2/#7/#10 substantially. **Pre-RoPE
>   becomes an OPTIONAL quality lever** (KVQuant per-channel outliers only — *not*
>   accumulation): it now costs an **explicit un-rotate at demotion** (reuse
>   `rerotate_keys`' un-rotate half) + a **per-step re-rotate at read** to the window's
>   original positions. The old "quantize between strip and re-rotate" piggyback
>   returns **only** if someone sets `rerotate_on_evict=True` (StreamingLLM-style).
> - **#3 collapses.** Positions are **never rebased** — each window keeps its true
>   original positions forever. No merged-`arange`, no per-tier slice reassignment, no
>   "positions change at compaction." The "one logical axis" is simply the **true
>   original positions** carried per window; relative distances are now **exact**, not
>   the old compaction approximation. The collision/independent-`arange` risk is gone.
> - **#7 cost drops** under the default post-RoPE Q tier: per step it's **dequant only**
>   (no RoPE), roughly halving the Q-tier read cost and removing the per-step-rotation
>   concern. (Pre-RoPE option re-adds rotation + the demotion strip.)
> - **#8 / #9 simplify.** `q_positions` never change ⇒ grids are trivially
>   position-invariant; the per-window record's position-slot is a **fixed original
>   range**, not a reassigned slice.
> - **G5 is moot** — the fp tier no longer re-rotates either, so neither tier has
>   rotation-compounding. (Removed.)
> - **Both backends already mirror this** (commit touched `windowed_cache` and
>   `windowed_eager_cache` symmetrically), consistent with #10.
>
> The affected resolutions below (#1, #2, #3, #7, #8, #9, #10) have been updated
> inline to match this amendment; no conflicting guidance remains.

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
   or a **Hadamard rotation** to spread them (RotateKV/QuaRot). Use **asymmetric**
   quant for the skewed distributions; validate **int8 first**, then int4.
6. **#11 — Tier-aware budget via a quant ratio.** Add a `quant_ratio` knob `q`
   that splits the **memory** budget (not the window count) between tiers:
   `M_budget = β·M_full`, `M_fp = (1−q)·M_budget`, `M_q = q·M_budget`. Convert each
   tier's memory to windows with **its own** bytes-per-window:
   `N_fp = M_fp / b_fp`, `N_q = M_q / b_q` where
   `b_q ≈ ¼·b_fp + per-window key-scale/zero + per-token value-scale overhead`.
   The int4 tier holds ~4× the windows of equal fp memory (minus overhead) — the
   resolver **must use `b_q`, not `b_fp`, for the Q tier**. Example (β=0.25, q=0.5):
   12.5% fp + 12.5% int4 ⇒ N_q≈4·N_fp ⇒ ~62.5% of windows at 25% memory. **Sink +
   local windows stay fp inside `M_fp`** (`top-K-fp = N_fp − (sink + local)`). Expose
   `β`, `q`, bit-width, group size as config knobs.
7. **#1 — Two dense stores, not a zero-padded tensor.** An **fp store**
   (`[B,H_kv,T_fp,D]` + `position_ids`) and a separate **Q store** (int4 codes +
   scales/zeros + `position_ids`), both **gap-free**. *Rejected:* a full-length fp
   tensor with zeros in Q slots — wastes memory and zero keys aren't softmax-neutral
   (`exp(q·0)=1`). RoPE needs only `position_id`, not co-location: each key
   carries its true original position (post-RoPE default ⇒ both tiers already rotated),
   so just **concatenate `[fp ‖ Q]`** (order-free during decode). Shared
   layout is a **logical index/tier map** (the per-window record, #9), not a tensor.
8. **#2 — Q-tier RoPE is a fork; default POST-RoPE (per AMENDMENT).** Rerotation is
   off by default, so there is no free strip gap. **Default: quantize the already-
   rotated surviving keys as-is** (post-RoPE) at demotion — **no strip**; at read
   **dequantize only, no rotation** (positions never change); promotion is a plain
   dequant. **Pre-RoPE is an OPTIONAL quality lever** (KVQuant per-channel outliers
   only — accumulation is already avoided by the no-rerotation architecture): it costs
   an **explicit un-rotate at demotion** (reuse `rerotate_keys`' un-rotate half) + a
   **per-step re-rotate at read** to the window's original positions. The old
   "quantize between strip and re-rotate" piggyback applies **only** under
   `rerotate_on_evict=True`. `update()` still returns one normal fp tensor; values
   carry no RoPE (asymmetric store).
9. **#3 — Original positions, never rebased (per AMENDMENT).** Rerotation off ⇒
   `slice_and_keep` keeps each surviving token's **true original `position_id`**;
   positions are **never** rebased to `arange` and never reassigned. So there is **no
   merged-`arange`, no per-tier slice juggling, no collision risk** — both tiers carry
   true positions, relative distances are **exact** (not the old compaction
   approximation), and the query keeps its absolute `cache_position`. The Q store
   records each window's fixed original position range (only needed at read **if** the
   pre-RoPE option is used). New tokens append at their natural absolute position.
10. **#7 — Per-step Q-tier cost: accepted, mitigated, measured.** Recent/local + sink
    + top-K stay fp, so the most-attended tokens skip the slow path. **Under the
    default post-RoPE Q tier the per-step cost is dequant-only (no RoPE)** — roughly
    half of the original estimate; the pre-RoPE option re-adds per-step rotation. A
    **benchmark gate**: Suite C (`perf_runner.py`) must confirm peak-memory savings
    aren't eaten by throughput/TPOT loss. (SKVQ-style recent-window-in-fp.)
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
    Hysteresis optional. **transformers 4.47.1 target across devices** (env follow-up:
    bump `environment.yml` pin from `<4.46`). ⚠️ *See "Environment caveat" below — the
    current dev machine actually runs 5.8.1.*
14. **#10 — Mirror the shared twins, keep hooks divergent.** "Mirror" = the
    byte-identical `cache.py`/`state.py` (+ new shared quant module), **not**
    `hooks.py` (flash recomputes via aux SDPA; eager reads materialized weights — we
    do **not** add aux SDPA to eager). Shared `update()` returns effective K/V
    `[fp ‖ dequant Q]` (post-RoPE default — dequant only; `+rotate` only under the
    pre-RoPE option), so eager scoring needs no change; the flash aux SDPA sources the
    same effective K via a shared `materialize_effective_kv` helper.
    Transient dequant is **per-layer** (modest, ~`(Q-fp size)/num_layers`).

### Locked decisions (bigger-picture review)
- **Full bidirectional promotion in v1** (per the initial prompt). Accept the
  #9/#12/#13 bookkeeping and the score-feedback risk (G4); **instrument promotion
  frequency + Suite A Jaccard-vs-fp-only over long sequences**. Documented fallback:
  one-way demotion + frozen Q-scores (not chosen).
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

## Kernel roadmap (three phases)

### Phase 1 — v1: materialize-then-concat (ship first)
Dequantize the entire Q store to fp16, concatenate with the fp store
`[fp ‖ dequant-Q]`, and pass the result to the standard attention path unchanged.
No custom kernels; fully CPU-testable; correctness is the only goal here.
- **Q-tier RoPE:** post-RoPE default — keys are stored already-rotated, so the
  dequantized tensor is immediately ready for attention with no further rotation.
- **Memory:** one transient fp16 copy of the Q tier per layer per step (freed after
  attention). Per-layer scope means peak impact ≈ `(Q-fp size)/num_layers` — modest.
- **Exit criterion:** Suite C confirms memory savings; Suite A Jaccard holds; LongBench
  quality acceptable at int8, then int4. Only then move to Phase 2.

### Phase 2 — Triton GEMV tile: fused dequant-inside-attention (future work)
A Triton decode kernel that loads int4 codes tile-by-tile, dequantizes to fp16
**in registers**, and runs `Q·K^T` before any write-back to global memory. The
fp16 materialization is eliminated entirely — only int4 codes are read from HBM.
- **Default: post-RoPE.** Keys are stored post-RoPE (same as Phase 1), so the tile
  kernel is: `load int4 → unpack → scale → MAC`. No in-kernel rotation. Straightforward
  to write (~150–200 lines Triton); directly removes the fp16 write-back cost.
- **Optional: pre-RoPE.** As a quality lever (KVQuant per-channel outlier benefit),
  keys can be stored pre-RoPE (un-rotated at demotion once; original positions cached
  as fixed `cos/sin` tables). The tile kernel becomes: `load int4 → unpack → scale →
  apply RoPE from cached cos/sin → MAC`. RoPE is pure arithmetic on already-loaded
  data — **zero extra memory traffic** in the bandwidth-bound decode regime; the
  arithmetic overhead is negligible. Pre-computed `cos/sin` tables are valid forever
  (positions never rebased). This option is **not in the initial kernel** — add only
  after the post-RoPE tile is validated and profiled.
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
