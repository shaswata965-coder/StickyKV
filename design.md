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
   (`exp(q·0)=1`). RoPE needs only `position_id`, not co-location: rotate each tier
   independently and **concatenate `[fp ‖ Q]`** (order-free during decode). Shared
   layout is a **logical index/tier map** (the per-window record, #9), not a tensor.
8. **#2 — Quantize between the strip and the re-rotate (reuse KVPress).** Does *not*
   break HF RoPE (pure function of `position_ids`; `update()` returns one normal
   rotated fp tensor). At eviction: **strip RoPE off all retained keys once**, then
   branch — top-K re-rotate→fp; top-Q **quantize the un-rotated pre-RoPE keys there**
   + store codes/position. Clean pre-RoPE grids (KVQuant). *Caveat:* still must
   **dequantize + rotate the Q tier at read each step** (the #7 cost). Values carry
   no RoPE (asymmetric store).
9. **#3 — One logical position axis, stored as per-tier slices.** Positions change
   **only at compaction**. *Rejected:* independent `arange`s per store (collide). At
   each eviction assign **one merged `arange(T_fp + T_q)`** over all retained tokens
   in chronological order, then slice into `fp_positions` / `q_positions`
   (interleaved, not contiguous). The fp tier is post-RoPE so needs no read-time
   positions; only the Q tier carries `q_positions`. New tokens append at running
   length; query position = current total length.
10. **#7 — Per-step Q-tier dequant cost: accepted, mitigated, measured.** Recent/
    local + sink + top-K stay fp, so the most-attended tokens skip the slow path;
    only the lower-weight Q tier is dequant+rotated per step. A **benchmark gate**:
    Suite C (`perf_runner.py`) must confirm peak-memory savings aren't eaten by
    throughput/TPOT loss. Fuse unpack+dequant+RoPE. (SKVQ-style recent-window-in-fp.)
11. **#8 — Gather is a non-issue; grids are position-invariant.** Start with
    **unpacked int8 codes** → `torch.gather` works token-wise; when packing to
    nibbles later, switch the Q store to **whole-window block selection**. The grid
    is computed from **pre-RoPE** keys (position-invariant), so position rebasing at
    eviction leaves codes + scale/zero valid; **only `q_positions` update**. Key the
    grid to window identity, never position.
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
    `[fp ‖ dequant+rotated Q]`, so eager scoring needs no change; the flash aux SDPA
    sources the same effective K via a shared `materialize_effective_kv` helper.
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
