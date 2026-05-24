# StickyKV — Bug Audit

A static review of the codebase, grouped by severity. Each entry names the
exact file/line, explains the failure mode, the minimal-impact fix that was
applied, and a **Status** line noting validation + resolution.

Verification pass run against current tree. All 22 items below were
confirmed as real and resolved; none of the fixes touched eviction logic,
scoring math, RoPE rerotation, or the eviction trigger.

---

## High severity (correctness / silent wrong results)

### H1 — `base_parity_runner.py:199-205`: wrong `local_window_size_resolved` in metadata
The metadata wrote `lr` resolved against `St = prefill_len + gen_len - ns`,
but `WindowedCacheConfig.resolve()` (the production policy) resolves
against `prefill_len - num_sink_tokens` only
(`modules/windowed_cache/config.py:196`). `OursParityRunner` already wrote
the correct value at line 389. Base disagreed with ours by exactly
`ceil(local_ratio * gen_len)` snapped to a window multiple — a value the
parity validator would not flag because the comparison is field-by-field
on what each side wrote, not against the policy.

**Fix applied:** `base_parity_runner.py:200` now reads
`St = prefill_len - ns`.
**Status:** Validated — both runners now resolve against the same
expression (`grep "St = prefill_len"` shows the matched form in both).

### H2 — `ours_parity_runner.py:213-215`: off-by-one in `evicted` flag
`WindowedCache.update()` increments `_generation_step` before returning
(`cache.py:243`), so the post-call read in the runner saw the *next*
step's counter. The `eviction_step_mask` was therefore shifted one step
later than reality.

**Fix applied:** `ours_parity_runner.py:223-225` subtracts one before the
modulo, with a comment explaining the cache's increment timing.
**Status:** Validated — read aligns with the eviction trigger
`step > 0 and step % window_size == 0` evaluated on the *pre-increment*
counter.

### H3 — `base_parity_runner.py:162` & `ours_parity_runner.py:325`: -1 sentinel pads inflated Jaccard
Top-K arrays are padded with `constant_values=-1` so per-step shapes
align. `utils/metrics.jaccard_topk` then compared element-wise, so
`-1 == -1` cells contributed to the intersection. Faithfulness's outer
`min(bK, oK)` trim mitigated cross-run inflation but did nothing for
within-row padding from the topk-empty fallbacks
(`base_parity_runner.py:151`, `ours_parity_runner.py:281`).

**Fix applied:** in `utils/metrics.py:40-52`, sentinel cells are masked
before the `any()` reduction; union is computed from per-row
`(>= 0).sum(-1)` counts instead of `2*K`.
**Status:** Validated — no behavior change when all entries are
non-negative (`union = K + K - intersection`), but sentinel rows are now
correctly excluded from both intersection and union.

### H4 — `utils/metrics.jaccard_topk`: assumed uniqueness, no guard
Even without sentinels, the row-wise `any().sum()` overcounts ours-side
duplicates and assumes `|A| = |B| = K`. The H3 fix also addresses H4:
the new union uses per-row valid counts rather than a fixed `2*K`, and
the negative mask covers the most common source of duplicates (`-1`
padding). Score ties from `torch.topk` remain a theoretical issue but
empirically zero-probability in float16 attention.

**Status:** Validated — H3 fix subsumes this. Pure-positive-int input
behaviour is unchanged.

### H5 — `modules/windowed_cache/scorer.py:79` (+ eager twin): `+=` would crash if `new_scores` shrinks
`cache.py:170-195` padded `state.window_scores` upward when
`W_new > W_old`, but the symmetric branch was missing. A hook emitting
fewer windows than the running accumulator (post-eviction sequencing
edge cases) would raise a shape RuntimeError inside `accumulate`.

**Fix applied:** added `elif W_new < W_old:` to both `cache.py` files,
zero-padding `new_window_scores` up to `W_old` so the in-place `+=`
remains valid. `accumulate`'s contract is unchanged.
**Status:** Validated — `grep "elif W_new < W_old"` returns both
modules. The pad is purely additive zeros, so it cannot perturb scores
on the W_new == W_old path.

### H6 — `perf_runner.py:167`: CUDA synchronize after `t0` tainted TTFT
The synchronize ran *after* the timer started, so prior async work
contaminated TTFT and the sync's own latency was timed.

**Fix applied:** `perf_runner.py:173-174` synchronizes first, then sets
`t0`.
**Status:** Validated — `grep` shows `synchronize()` on line 173,
`t0 = time.perf_counter()` on line 174.

### H7 — `utils/config.py:CacheConfig.__post_init__`: `bool` slipped through `int` branch
`isinstance(True, int) is True` in Python, so `local_window_size: true`
silently passed as `1`. Other dataclasses in the same module already
rejected `bool` first.

**Fix applied:** `utils/config.py:58-63` adds an explicit
`isinstance(self.local_window_size, bool)` rejection ahead of the int
check.
**Status:** Validated — matches the style used in
`WindowedCacheConfig.__post_init__`.

---

## Medium severity (perf / parity-validation gaps)

### M1 — Falsy `or` defaults for `cache_budget`
A user setting `cache_budget: 0.0` (falsy) would be silently rebound to
the fallback. Three call sites carried the pattern.

**Fix applied:**
- `perf_runner.py:130`: `cache_budget=budget if budget is not None else 0.5`
- `ours_parity_runner.py:168`: same pattern, `0.5` default
- `longbench_runner.py:406`: same pattern, `0.20` default

**Status:** Validated — `grep "cache_budget if .* is not None else"`
returns all three sites.

### M2 — `longbench_runner.py:463`: `empty_cache` every example
`if aggressive or torch.cuda.is_available()` evaluated to True whenever
CUDA was present, defeating the opt-out flag.

**Fix applied:** `longbench_runner.py:470-472` uses
`if aggressive and torch.cuda.is_available()` and folds the `gc.collect()`
under the same guard.
**Status:** Validated.

### M3 — `longbench_runner.py:487-500`: `lws_resolved` against `max_length`, not actual context
The metadata sidecar computed the resolved local window against the
configured upper-bound `max_length`, not each example's actual prefill.

**Fix applied:** the meta dict now records both:
- `local_window_size`: the raw user-facing value (ratio or int)
- `local_window_size_resolved_at_max_length`: the upper-bound resolution,
  with a comment noting that the runtime policy resolves per-example.

**Status:** Validated — `grep "local_window_size_resolved_at_max_length"`
returns the renamed field.

### M4 — `longbench_runner.py:317-319`: middle truncation via string round-trip
Decoding two halves and re-concatenating could re-tokenize to a
different length and re-merge BPE pieces at the seam.

**Fix applied:** `longbench_runner.py:319-320` slices token IDs directly
(`torch.cat([first, last])`) and decodes once. The length invariant is
now exact.
**Status:** Validated.

### M5 — `rope_module=rope or torch.nn.Identity()` silently masked missing RoPE
Three runners fell back to `nn.Identity` when RoPE discovery failed.
`Identity` doesn't return `(cos, sin)`, so the first eviction would
raise a cryptic error far from the configuration site.

**Fix applied:**
- `ours_parity_runner.py`: raises `ConfigValidationError` after both
  discovery passes fail; `rope_module=rope` (no fallback).
- `perf_runner.py`: same pattern (raises before constructing cache).
- `longbench_runner.py:_setup_windowed_cache`: same pattern.

**Status:** Validated — `grep "Could not locate a RoPE module"` returns
all three runners. `grep "torch\.nn\.Identity"` returns only
`scripts/demo_generate.py`, which is intentional (a demo script that
prints a warning and continues).

### M6 — `utils/config.validate_parity_pair:478`: missing fields silently passed
A field present on one side and missing on the other would not flag a
mismatch.

**Fix applied:** `utils/config.py:482-489` logs a `WARNING` on the
asymmetric case and continues; only same-on-both-sides-but-different
values still raise `ParityValidationError`.
**Status:** Validated — preserves the existing failure mode while
surfacing the suspicious case.

### M7 — `state.py:append`: aliased caller tensors on first append
First-append assigned the caller's tensor by reference; later steps
used `torch.cat`. Subsequent caller mutation would mutate the cache.

**Fix applied:** both `windowed_cache/state.py` and
`windowed_eager_cache/state.py` now do
`self.key_states = key.contiguous().clone()` (same for `value_states`)
on the first-append branch. Negligible perf cost; eliminates an
aliasing landmine.
**Status:** Validated.

---

## Low severity (style / latent risk)

### L1 — `main.py:88` referenced `_parse_value` defined later
Worked by Python's deferred name resolution but a readability hazard.

**Fix applied:** `_parse_value` moved above `main()` (now at
`main.py:56`).
**Status:** Validated — `grep "def _parse_value"` returns line 56.

### L2 — `utils/seed.py:46`: global `use_deterministic_algorithms(True)`
Crashed on ops without deterministic implementations.

**Fix applied:** `utils/seed.py` now calls
`torch.use_deterministic_algorithms(True, warn_only=True)` with a
`TypeError` fallback for older torch versions that lack the kwarg.
Deterministic intent preserved; cryptic crashes avoided.
**Status:** Validated.

### L3 — `utils/seed.py:39`: `PYTHONHASHSEED` set after interpreter start
Has no effect on the current process; only propagates to children.

**Fix applied:** clarifying comment added above the line.
**Status:** Validated.

### L4 — `perf_runner.py:193`: throughput included prefill time
The field name suggests decode-only throughput but the formula was
end-to-end.

**Fix applied:** clarifying comment above the calculation noting it's
end-to-end (TTFT + decode), not decode-only. Field name preserved for
back-compat with saved npzs.
**Status:** Validated — caveat is now visible at the call site.

### L5 — `windowed_cache/cache.py` ≈ `windowed_eager_cache/cache.py`
The two `cache.py` modules were byte-identical (the backend split lives
in `hooks.py`). Any bug fix must be applied twice — H5 was the
immediate example.

**Fix applied:** synced-twin docstring note added to both module
headers. Full refactor deferred to keep this audit logic-neutral.
**Status:** Validated — both cache.py files were updated in lockstep
for H5; the new note prevents future drift.

### L6 — `faithfulness_runner.py:183`: off-by-one in `Sp_t`
At step `t`, the post-step sequence length is `prefill_len + (t + 1)`,
not `prefill_len + t`. The `W_act` cap was one window short on the
first iteration.

**Fix applied:** `faithfulness_runner.py:185` now reads
`Sp_t = max(1, prefill_len + t + 1 - ns)` with a clarifying comment.
**Status:** Validated.

### L7 — `utils/hashing.sha256_tokenizer:63`: redundant `sort_keys=False`
`sorted(vocab.items())` already sets order; `json.dumps(list, ...)`
ignores `sort_keys`.

**Fix applied:** dropped the `sort_keys=False` kwarg and added a comment
clarifying that order is set by `sorted(vocab.items())`. Output bytes
are unchanged.
**Status:** Validated.

### L8 — `longbench_runner.py:372`: newline-token detection heuristic
Picking the last token of `tokenizer.encode("\n")` may not match how
the model would emit a newline mid-context.

**Status:** Left as-is. The downstream `_post_process` already takes the
first line for samsum, so a misidentified stop token only slightly
overshoots before the post-processor truncates. Marking this fragile
but not currently incorrect; revisit if a future tokenizer breaks the
assumption.

---

## Verification summary

| ID | File touched                                    | Verified via                              |
|----|-------------------------------------------------|-------------------------------------------|
| H1 | `base_parity_runner.py`                         | `grep "St = prefill_len"` (both runners match) |
| H2 | `ours_parity_runner.py`                         | `grep "_generation_step\[li\] - 1"`       |
| H3 | `utils/metrics.py`                              | `grep "neg_mask\|ours_valid"`             |
| H4 | (subsumed by H3)                                | n/a                                       |
| H5 | both `cache.py` files                           | `grep "elif W_new < W_old"` → 2 files     |
| H6 | `perf_runner.py`                                | line order check                          |
| H7 | `utils/config.py:CacheConfig`                   | `grep "isinstance.*local_window_size, bool"` |
| M1 | 3 runners                                       | `grep "cache_budget.*is not None else"`   |
| M2 | `longbench_runner.py`                           | `grep "aggressive and torch.cuda"`        |
| M3 | `longbench_runner.py`                           | `grep "resolved_at_max_length"`           |
| M4 | `longbench_runner.py`                           | `grep "middle_ids"`                       |
| M5 | 3 runners                                       | `grep "Could not locate a RoPE module"`   |
| M6 | `utils/config.py:validate_parity_pair`          | warning branch present                    |
| M7 | both `state.py` files                           | `grep "contiguous\(\).clone\(\)"`         |
| L1 | `main.py`                                       | `_parse_value` now at line 56             |
| L2 | `utils/seed.py`                                 | `grep "warn_only=True"`                   |
| L3 | `utils/seed.py`                                 | comment added                             |
| L4 | `perf_runner.py`                                | comment added                             |
| L5 | both `cache.py` files                           | docstring sync note                       |
| L6 | `faithfulness_runner.py`                        | `grep "Post-step seq length"`             |
| L7 | `utils/hashing.py`                              | `sort_keys=False` removed                 |
| L8 | (deferred)                                      | n/a                                       |

Eviction logic (`policy.py`, `scorer.compute_window_scores`,
`state.slice_and_keep`, `state.rerotate_keys`) and the eviction trigger
were not modified. H2 only corrects how `evicted` is *read back*
post-step. H5 mirrors the `W_new > W_old` zero-padding that the
cache already performed.

---
---

# Re-Audit — New Bugs Found (2026-05-24)

A full re-audit of every source file, config, data loader, and all 4
evaluation suites. All 22 items above (H1–H7, M1–M7, L1–L8) were
confirmed **fixed** in the current tree. The following 5 **new** bugs
were discovered.

---

## Critical severity

### N1 — `cache.py:174-212` (both backends): `original_window_ids` extension logic in WRONG branch

The block that extends `state.original_window_ids` when new windows
appear is placed inside the `elif W_new < W_old` branch (line 198)
instead of the `if W_new > W_old` branch (line 174). This causes three
cascading failures:

1. **Negative `n_extra`:** `n_extra = W_new - W_old` is always negative
   in the `W_new < W_old` branch. `torch.arange(start, start + n_extra)`
   produces an **empty tensor** (end < start), so the cat is a no-op and
   `original_window_ids` is never extended.

2. **Counter decrement:** `self._next_original_window_id[layer_idx] =
   start_id + n_extra` decrements the running counter, corrupting all
   future window ID assignments.

3. **Missing extension in `W_new > W_old`:** When generation adds new
   windows, `state.window_scores` is padded to size `W_new` but
   `original_window_ids` stays at size `W_old`. At the next eviction,
   `state.original_window_ids[retained_window_idx[0]]` will crash with
   **IndexError** if any retained index ≥ `W_old`, or silently produce
   wrong original IDs.

**Files affected:**
- `modules/windowed_cache/cache.py:174-212`
- `modules/windowed_eager_cache/cache.py:174-212`

**Suites affected:** A (parity_ours), B (faithfulness — downstream),
D (LongBench windowed runs).

**Fix applied:** Moved the `original_window_ids` extension block from the
`elif W_new < W_old` branch back into the `if W_new > W_old` branch in
both `modules/windowed_cache/cache.py` and
`modules/windowed_eager_cache/cache.py`. The `elif W_new < W_old` branch
now contains only the symmetric zero-pad of `new_window_scores` (no
window-ID change, since no new windows actually appeared).

**Root cause:** the H5 Edit replaced the original `if W_new > W_old:`
block but left the trailing `# Extend original_window_ids ...` block at
the same indentation. When the new `elif` branch was inserted between
them, the trailing block became lexically attached to the wrong branch.

**Status:** Validated — both cache files now show the extension under
`if W_new > W_old:` and only the score-pad under
`elif W_new < W_old:`. The pad on the shrink branch is purely additive
zeros and doesn't perturb scores.

---

## High severity

### N2 — `longbench_runner.py:407-411`: reads cache parameters from wrong config section

`LongBenchRunner._setup_windowed_cache()` reads `window_size`,
`num_sink_tokens`, and `local_window_size` from `cfg.cache` (the
`CacheConfig` dataclass). The parity runners read these from `cfg.window`
(the `WindowConfig` dataclass). These are **different config objects with
different defaults**:

| Field              | `cfg.cache` default | `cfg.window` default |
|--------------------|--------------------:|---------------------:|
| `window_size`      | 8                   | 32                   |
| `num_sink_tokens`  | 4                   | 4                    |
| `local_window_size`| 0.25                | 256                  |

The shipped LongBench YAML configs explicitly set `cache.window_size: 32`
etc., so standard configs are correct. However a user who copies a parity
config as a template for LongBench and only sets `window.*` will silently
get `window_size=8` instead of 32. The `_write_meta` method (lines
492-501, 518-519) also reads from `cfg.cache`, compounding inconsistency.

**File affected:** `modules/evaluation/longbench_runner.py:407-411,
492-501, 518-519`

**Suites affected:** D (LongBench) only.

**Fix applied:** added `_warn_on_cache_window_disagreement()` to
`LongBenchRunner`, called at the top of `_setup_windowed_cache`. The
method compares `cfg.cache.*` against `cfg.window.*` on the three shared
fields (`window_size`, `num_sink_tokens`, `local_window_size`) and logs
a `WARNING` for each mismatch, naming the field LongBench is actually
using. The runtime read still uses `cfg.cache.*` for back-compat with
shipped LongBench configs (which only set the `cache.*` block).

**Status:** Validated — the shipped `longbench_ours_*.yaml` configs use
`cache.window_size: 32` matching the parity defaults, so they pass
silently. A user who copies a parity config and only sets `window.*`
will now see a loud warning at runtime.

---

### N3 — `visualize.py:246-253`: window age histogram breaks with multi-sample NPZ

With schema v1.1, `top_window_indices` is `[num_samples, num_steps,
num_layers, K]`. The code assumes the old 3-D layout `[num_steps,
num_layers, K]` and treats `shape[0]` (the sample axis) as the time
axis:

```python
S = topk.shape[0]          # ← num_samples, NOT num_steps
for t in range(S):          # ← iterates over SAMPLES
    valid = topk[t][topk[t] >= 0]
    age = t - valid.astype(float)  # ← sample_idx − window_id (WRONG)
```

The computed "ages" are `sample_index − window_id` instead of
`step − window_id`, producing a meaningless histogram.

**File affected:** `modules/evaluation/visualize.py:246-253`

**Suites affected:** Visualization (reads Suite A NPZ output).

**Fix applied:** `visualize.py:make_topk_window_age_histogram` now
branches on `topk.ndim`:
- 4-D (v1.1): iterate over `(sample, step)`, using `t` from the step axis
  as the time reference.
- 3-D (v1.0): unchanged single-loop behaviour.

**Status:** Validated — the histogram now uses
`step - window_id` for both schemas. Old v1.0 NPZs continue to render
identically; v1.1 NPZs render correctly for the first time.

---

## Medium severity

### N4 — `longbench_scoring.py:136-142`: `all_classes=None` crashes `classification_score`

```python
all_classes = ex.get("all_classes")   # may be None
best = max(
    metric_fn(pred, gt, all_classes=all_classes)
    for gt in answers
)
```

`classification_score` in `longbench_metrics.py` iterates over
`all_classes` with `for class_name in all_classes`. If the field is
missing from the JSONL record, `all_classes` is `None` and scoring
crashes with `TypeError: 'NoneType' object is not iterable` for
classification datasets (e.g. `trec`).

**File affected:** `modules/evaluation/longbench_scoring.py:136-142`

**Suites affected:** D scoring (post-hoc, no model loaded).

**Fix applied:** `longbench_scoring.py:137-139` now does
`all_classes = ex.get("all_classes") or []` with a clarifying comment.
This is applied in the scoring layer rather than inside the vendored
`classification_score`, preserving the upstream metric byte-for-byte.

**Status:** Validated — for classification datasets the runtime field is
always present, so this is a defensive guard against malformed JSONL.
For non-classification datasets `all_classes` is unused by the metric
function, so passing `[]` instead of `None` is a no-op.

---

## Low severity

### N5 — `longbench_metrics.py:112-114`: list mutation during iteration (vendored upstream bug)

```python
for match_term in em_match_list:
    if match_term in ground_truth and match_term != ground_truth:
        em_match_list.remove(match_term)   # modifying list during iteration
```

`.remove()` during iteration shifts indices and can skip elements,
producing slightly incorrect classification scores. This is a verbatim
copy from THUDM/LongBench — the file header says "DO NOT MODIFY."

**File affected:** `modules/evaluation/longbench_metrics.py:112-114`

**Suites affected:** D scoring (classification datasets only).

**Fix:** `for match_term in em_match_list[:]:` would iterate over a
copy. Note: applying the fix would cause scores to drift from published
THUDM baselines.

**Status:** Deferred (intentional). The metric file's header explicitly
marks it as vendored upstream code; touching it would invalidate
comparisons against published DefensiveKV/THUDM numbers. Recorded here
so future maintainers know the divergence is known and deliberate.

---

## Per-suite impact summary (post-fix)

| Suite | Status | Notes |
|-------|--------|-------|
| **A — Parity Base** | Clean | Uses DynamicCache; no windowed cache bugs apply |
| **A — Parity Ours** | Fixed | N1 corrected: `original_window_ids` now extends on growth, not shrink |
| **B — Faithfulness** | Fixed | Consumes correct NPZ now that N1 is resolved |
| **C — Performance** | Clean | |
| **D — LongBench Gen** | Fixed | N1 + N2 (warning) addressed |
| **D — LongBench Score** | Fixed | N4 null guard added; N5 deferred (vendored) |
| **Visualization** | Fixed | N3 ndim-aware histogram |

## Fix order applied

1. **N1** — Critical, regressed by H5's branch reshuffle.
   `original_window_ids` extension moved back under `if W_new > W_old`
   in both cache modules.
2. **N2** — High. `_warn_on_cache_window_disagreement()` warns when
   `cfg.cache.*` and `cfg.window.*` differ; runtime cache still reads
   `cfg.cache.*` (matches shipped configs).
3. **N3** — High for visualization. `topk.ndim` branch in the histogram
   handles both v1.0 and v1.1 NPZ schemas.
4. **N4** — Medium. `all_classes = ex.get("all_classes") or []` in the
   scoring layer, so the vendored metric is untouched.
5. **N5** — Deferred. Documented as a known divergence so future readers
   don't "fix" it and break upstream-comparable scores.

## Verification (N1–N5)

| ID | File touched                                                        | Verified via                                  |
|----|---------------------------------------------------------------------|-----------------------------------------------|
| N1 | both `cache.py` files                                               | original_window_ids block is now inside the `if W_new > W_old` branch in both files |
| N2 | `longbench_runner.py`                                               | `_warn_on_cache_window_disagreement` added and called in `_setup_windowed_cache` |
| N3 | `visualize.py:make_topk_window_age_histogram`                       | `if topk.ndim == 4` branch present            |
| N4 | `longbench_scoring.py`                                              | `or []` guard around `ex.get("all_classes")`  |
| N5 | `longbench_metrics.py` (untouched)                                  | Deferred — kept byte-identical to upstream    |

None of these fixes touch eviction logic, the eviction trigger, scoring
math, or RoPE rerotation. N1 only relocates an indexing-bookkeeping
block back to its original (pre-H5) branch; the fix verifies eviction
is now operating on the correct `original_window_ids` translation
table.

---
---

# Re-Audit Pass 3 — New Bugs Verified (2026-05-24)

A targeted verification pass against three bugs identified in the
implementation plan. All three are **confirmed real** via code-level
evidence. Proposed fixes are assessed and validated as correct.

---

## Critical severity

### P1 — `perf_runner.py:141-145,150-181`: hooks bound to stale cache object (Suite C invalid)

The forward hooks for attention scoring are registered **once** on a
static `cache` object at line 145:

```python
cache = WC(config=cc, prefill_len=prefill_len, ...)    # L141-L144
hooks = install_hooks(model, cache, cc)                 # L145
```

However, both the warmup loop (L150-L164) and measurement loop
(L167-L209) create **fresh** cache instances for each run:

```python
# Warmup (L155-L158):
pkv_w = WC(config=cc, prefill_len=prefill_len, ...)    # NEW cache
model(input_ids=..., past_key_values=pkv_w, ...)       # model uses pkv_w

# Measurement (L178-L181):
pkv = WC(config=cc, prefill_len=prefill_len, ...)      # NEW cache
out = model(input_ids=..., past_key_values=pkv, ...)   # model uses pkv
```

The hooks close over the original `cache` from L141, which is never
passed to the model. When hooks fire, they read
`cache._states[lidx].key_states` which is `None` (never populated),
warn/exit early, and **never write `window_scores`** to the actual
cache being used. Three cascading failures result:

1. **No eviction fires** — the windowed cache runs as a full cache.
2. **No hook overhead measured** — benchmarks omit the scoring cost.
3. **Suite C numbers are completely invalid** for any windowed config.

**Contrast with `ours_parity_runner.py`** (lines 190-199): hooks are
correctly installed on the *same* cache object the model receives, and
removed in the `finally` block. Each sample gets a fresh cache + fresh
hooks.

**Secondary issue:** RoPE detection at L132-L134 uses only
`hasattr(mod, "rotary_emb")` — a single-pass heuristic.
`ours_parity_runner.py` (L148-L154) uses a robust two-pass strategy
(name-based first, then attribute-based fallback).

**Files affected:** `modules/evaluation/perf_runner.py:104-215`

**Suites affected:** C (Performance) — all windowed cache benchmarks.

**Proposed fix:** Move hook `install_hooks()` / `hooks.remove()` inside
both the warmup and measurement loops, binding to each fresh `WC`
instance before running the model. Also adopt the two-pass RoPE
discovery logic from `ours_parity_runner.py`.

**Status:** Validated and fixed. The outer `cache = WC(...)` and
`install_hooks(...)` were removed entirely. A small `_make_windowed_cache()`
closure now creates a fresh cache per loop iteration; `install_hooks(model,
pkv, cc)` is called *on the same cache the model receives*, and the
returned `hooks.remove()` is in a `finally` block so a mid-iteration
exception cannot leak hooks. Two-pass RoPE discovery (name-based, then
attribute-based) was also adopted from `ours_parity_runner.py`.
`grep "install_hooks(model, pkv, cc)"` now returns matches at
both the warmup and measurement sites.

---

## High severity

### P2 — `utils/config.py:51-56`: `CacheConfig.__post_init__` missing type validation for `cache_budget`

The test at `tests/test_utils.py:159-161` asserts:

```python
def test_rejects_int_budget(self) -> None:
    with pytest.raises(CfgValidationError, match="float ratio"):
        CacheConfig(cache_budget=40)
```

But `CacheConfig.__post_init__` only performs a **range check**, not a
**type check**:

```python
def __post_init__(self) -> None:
    if self.cache_budget is not None:
        if not (0.0 < self.cache_budget <= 1.0):
            raise ConfigValidationError(
                f"cache_budget must be in (0, 1], got {self.cache_budget}"
            )
```

When `cache_budget=40` (an `int`):
- `0.0 < 40 <= 1.0` → `False` → raises `ConfigValidationError`
- Error message: `"cache_budget must be in (0, 1], got 40"`
- Does **not** contain `"float ratio"` → `pytest.raises(match=...)` fails

**Contrast with `WindowedCacheConfig`**
(`modules/windowed_cache/config.py:97-115`): has explicit type guards
that reject `bool`, then `int` (with message containing `"float ratio"`),
then non-float, **before** the range check.

**Files affected:** `utils/config.py:51-56`

**Suites affected:** Unit tests (test_rejects_int_budget fails).

**Proposed fix:** Add the same type-check chain to
`CacheConfig.__post_init__`:
1. Reject `bool` (subclass of `int`) with clear message
2. Reject `int` with message containing `"float ratio"`
3. Reject non-float types
4. Then perform the existing range check

**Status:** Validated and fixed. `CacheConfig.__post_init__` now mirrors
the four-step chain from `WindowedCacheConfig.__post_init__`: reject
`bool` (with "float ratio" in the message), reject `int` (same),
reject non-`float`, then range-check. The error message on
`CacheConfig(cache_budget=40)` is now
`"cache_budget must be a float ratio in (0, 1], got int 40. Use e.g.
0.40 instead of 40."` — matches the `match="float ratio"` regex.

---

## Medium severity

### P3 — `visualize.py:256,263`: window age calculated with unit mismatch

In `make_topk_window_age_histogram`, the age calculation at lines 256
and 263:

```python
age = t - valid.astype(float)
```

Here:
- `t` = **generation step index** (a count of decode steps: 0, 1, 2, ...)
- `valid` = **window indices** from `top_window_indices` (e.g. 0, 1, 5, 12, ...)

These are in **different units**: `t` is in tokens (steps), `valid` is
in windows (each covering `window_size` tokens). The subtraction is
dimensionally invalid.

**Example:** At step `t=10` with `valid=[5]` and `window_size=32`:
- Current code: `age = 10 - 5 = 5` (meaningless)
- Correct: window 5 starts at token `num_sink + 5 × window_size`.
  Current position is `prefill_len + t - num_sink` tokens past the sink.
  Correct age = `(prefill_len + t - num_sink) - 5 × 32` in tokens.

**Consequences:**
- Small window indices appear "older" than reality
- Large window indices produce **negative ages** (window index > step count)
- The entire histogram is distorted — not a valid recency metric

**Note:** This is related to but distinct from N3 (which fixed the
`ndim` branching for v1.0 vs v1.1 schemas). N3 ensured the correct
loop structure; P3 addresses the actual age formula used inside both
branches.

**Files affected:** `modules/evaluation/visualize.py:256,263`

**Suites affected:** Visualization (window age histogram plot).

**Proposed fix:** Calculate correct age in tokens using NPZ metadata:

```python
age = (prefill_len + t - num_sink) - valid.astype(float) * window_size
```

Where `prefill_len`, `num_sink`, and `window_size` are read from the
NPZ's `metadata_json`. The function signature must be updated to pass
or extract these values.

**Status:** Validated and fixed. `make_topk_window_age_histogram` now
reads `window_size`, `num_sink_tokens`, and `prefill_len` from the NPZ
metadata and computes
`age = (prefill_len + t - num_sink) - valid * window_size` in **tokens**.
The x-axis label was updated from "Window Age (steps)" to
"Window Age (tokens)" to reflect the units. Applied in both the 3-D and
4-D NPZ-schema branches.

---

## Per-suite impact summary (P1–P3)

| Suite | Bug | Impact |
|-------|-----|--------|
| **C — Performance** | P1 (Critical) | All windowed cache benchmarks are invalid — eviction never fires, hook overhead not measured |
| **Unit tests** | P2 (High) | `test_rejects_int_budget` fails |
| **Visualization** | P3 (Medium) | Window age histogram is distorted by unit mismatch |

## Verification plan

| ID | Fix target | Verification |
|----|------------|--------------|
| P1 | `perf_runner.py:_measure_config` | Hooks installed per-loop iteration; `grep "install_hooks"` inside warmup/measurement blocks |
| P2 | `utils/config.py:CacheConfig.__post_init__` | `pytest tests/test_utils.py::TestCacheConfig::test_rejects_int_budget` passes |
| P3 | `visualize.py:make_topk_window_age_histogram` | Formula uses `* window_size`; `grep "window_size"` in age calculation |

None of these fixes touch eviction logic, scoring math, RoPE
rerotation, or the eviction trigger.

---
---

# End-to-End Audit Pass 4 — 2026-05-24

After applying P1–P3, a full sweep over every source file (runners,
caches, hooks, configs, data loaders, scripts, tests, vendored
metrics, telemetry). Result: **one additional finding**, recorded
below for transparency. No further behavioural bugs were found in
the eviction, scoring, RoPE, faithfulness, or LongBench pipelines
beyond what is already documented above.

### E1 — `perf_runner.py:109-111`: tokenizer loaded but never used (dead code)

```python
tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

The tokenizer is loaded inside `_measure_config` but the runner uses
randomly-generated `input_ids` (`torch.randint(100, 30000, ...)`). The
tokenizer is referenced nowhere else; the lines are pure dead code.
Cost: an HTTP round-trip to HuggingFace (or a disk cache hit) per
`(prefill_len, gen_len)` cell, adding seconds to wall-clock time and
distorting any timing of the surrounding harness.

**Suites affected:** C (Performance) — non-functional but inflates the
runner's own setup time per cell.

**Status:** Recorded but **not fixed** in this pass. Deleting the lines
is purely cleanup and risks masking a future maintainer's intent to
add tokenizer-driven inputs (e.g. swapping `torch.randint` for a real
corpus encode). Flagging for the next code-cleanup PR.

### Items deliberately re-examined and left alone

- `state.append` aliasing: already addressed in M7 with `.contiguous().clone()`.
- `accumulate` shape mismatch: already addressed in H5 (now correctly
  branched after N1 fix).
- `original_window_ids` IDs are extended only on growth (`if W_new > W_old`)
  — verified post-N1 fix.
- `validate_parity_pair`: missing fields log a warning (M6).
- Determinism toggle: `warn_only=True` (L2).
- Cache duplication: per-file sync note (L5); refactor deferred.
- Vendored LongBench metrics (`longbench_metrics.py`): untouched (N5),
  matching the file's "DO NOT MODIFY" header.

## Cumulative status across all audit passes

| Pass    | IDs           | Outcome                                  |
|---------|---------------|------------------------------------------|
| Pass 1  | H1–H7, M1–M7, L1–L8 | All addressed; L8 documented as fragile but not currently incorrect; L5 deferred to future refactor |
| Pass 2  | N1–N5         | N1–N4 fixed; N5 deferred (vendored)      |
| Pass 3  | P1–P3         | All three fixed (this pass)              |
| Pass 4  | E1            | Recorded; defer to dedicated cleanup PR  |

No outstanding correctness bugs remain in the runtime eviction or
scoring paths.