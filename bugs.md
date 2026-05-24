# StickyKV — Bug Audit

A static review of the codebase, grouped by severity. Each entry names the
exact file/line, explains the failure mode, and gives a minimal-impact fix
that preserves the existing logic.

---

## High severity (correctness / silent wrong results)

### H1 — `base_parity_runner.py:199-205`: wrong `local_window_size_resolved` in metadata
The metadata writes `lr` resolved against `St = prefill_len + gen_len - ns`,
but `WindowedCacheConfig.resolve()` (the production policy) resolves against
`prefill_len - num_sink_tokens` only (see `modules/windowed_cache/config.py:196`
and `utils/config.py:WindowConfig.resolved_top_k`). `OursParityRunner` writes
the *correct* value at line 389 (`St = prefill_len - ns`). The base run therefore
records a value that disagrees with the matching ours run.

**Fix:** replace `St = prefill_len + gen_len - ns` with `St = prefill_len - ns`
in `base_parity_runner.py:199`. No other logic touched; this only corrects the
recorded scalar.

### H2 — `ours_parity_runner.py:213-215`: off-by-one in `evicted` flag
The eviction probe runs *after* `model(...)`. By that point
`WindowedCache.update()` has already incremented `_generation_step[li]` (see
`cache.py:243`). So `cache._generation_step[li] % ws_sz == 0` is checking the
*next* step's eviction trigger, not the one that just fired. The
`eviction_step_mask` array is therefore shifted by one step and consistently
mislabels which steps actually evicted.

**Fix:** subtract one before the modulo:
```python
gs = cache._generation_step[li]
evicted = any((gs - 1) > 0 and (gs - 1) % ws_sz == 0 for li in range(n_layers))
```
or expose an `evicted_this_step` flag from `WindowedCache.update()` and read
that instead. Logic of eviction itself is unchanged; only the post-hoc
telemetry flag is corrected.

### H3 — `base_parity_runner.py:162` & `ours_parity_runner.py:325`: -1 sentinel pads inflate Jaccard
Top-K arrays are padded with `constant_values=-1` so that shapes align
across steps/samples. Downstream `utils/metrics.jaccard_topk` does
`ours_exp == base_exp` element-wise (see `metrics.py:40-44`); two `-1`
sentinels compare equal and contribute to the intersection. The
`FaithfulnessRunner` partially mitigates this by trimming to
`min(bK, oK)` (line 148-150), but **within-sample / within-step** padding
of `step_tk` (lines 149, 151, 281 of the ours runner) can still leave -1
entries inside the K-dimension.

**Fix (minimal):** in `utils/metrics.jaccard_topk` mask out negative
indices before the `any()` reduction:
```python
neg_mask = (ours_exp < 0) | (base_exp < 0)
matches = (ours_exp == base_exp) & ~neg_mask
```
Logic of Jaccard itself is preserved; this only stops sentinel collisions.

### H4 — `utils/metrics.jaccard_topk`: assumes uniqueness, no guard
Even without -1 sentinels, the `matches.any(dim=-1).sum(dim=-1)` formula
double-counts ours-side duplicates and assumes `|A| = |B| = K`. `torch.topk`
gives unique indices when there are no score ties, but H3's sentinels and
edge cases (e.g. `step_tk.append(np.zeros(...))` fallbacks at
`base_parity_runner.py:151` and `ours_parity_runner.py:281`) violate
uniqueness.

**Fix:** along with the negative-mask in H3, deduplicate per-row before the
match count, or compute the union explicitly as `(A_unique ∪ B_unique).numel()`.

### H5 — `modules/windowed_cache/scorer.py:79` (and the eager twin): in-place `+=` will crash if `new_scores` shrinks
`accumulate()` does `state_scores += new_scores`. `WindowedCache.update()`
only pads `state.window_scores` upward when `W_new > W_old`
(`cache.py:170-195`); the symmetric `W_new < W_old` branch is missing. A
late-step hook that happens to emit a smaller W (e.g. immediately after an
eviction collapses the window count and before new generation tokens
re-extend it) will hit a shape-mismatch RuntimeError.

**Fix:** handle the other branch in `cache.py` (zero-pad `new_window_scores`
up to `W_old`) before calling `accumulate`, mirroring the existing
W_new > W_old path. Same change required in the `windowed_eager_cache`
copy.

### H6 — `perf_runner.py:167`: CUDA synchronize **after** `t0` taints TTFT
```python
t0 = time.perf_counter()
if torch.cuda.is_available(): torch.cuda.synchronize()
```
Any prior asynchronous work that wasn't flushed will be counted toward
TTFT, but the launch overhead of synchronize itself is *also* timed.
The synchronize must run **before** `t0` (and ideally a no-op forward
should precede warmup) so TTFT measures only the prefill forward.

**Fix:** move the first `torch.cuda.synchronize()` immediately above
`t0 = time.perf_counter()`. No measurement logic changes; only the order
of two lines.

### H7 — `utils/config.py:CacheConfig.__post_init__`: `bool` slips through int branch
`isinstance(True, int)` is `True` in Python, so `local_window_size=True`
passes the `int` branch silently and downstream arithmetic treats it as `1`.
Other dataclasses in the same file explicitly reject `bool` first (e.g.
`WindowedCacheConfig.__post_init__` at lines 78, 88, 98); `CacheConfig`
forgot to.

**Fix:** add a `isinstance(self.local_window_size, bool)` guard before the
`isinstance(..., int)` check, matching the style used in
`modules/windowed_cache/config.py:118-119`.

---

## Medium severity (perf / parity-validation gaps)

### M1 — Falsy `or` defaults for `cache_budget`
- `perf_runner.py:130`: `cache_budget=budget or 0.5`
- `ours_parity_runner.py:161`: `budget = cfg.cache.cache_budget or 0.5`
- `longbench_runner.py:405`: `budget = cfg.cache.cache_budget or 0.20`

A user who explicitly sets `cache_budget: 0.0` (or any falsy value) is
silently rebound to the fallback. The intent is "if None, use default",
which should be `if budget is None`.

**Fix:** replace each with an explicit `None` check. Same control flow,
only the predicate changes.

### M2 — `longbench_runner.py:463`: `empty_cache` called every example
```python
if aggressive or torch.cuda.is_available():
    torch.cuda.empty_cache()
```
The `or` reads "aggressive cleanup OR CUDA present"; when CUDA is present
(the common case) this always fires, even when the config opted out.
`empty_cache()` per example measurably distorts throughput on long-running
LongBench runs.

**Fix:** `if aggressive and torch.cuda.is_available():`.

### M3 — `longbench_runner.py:487-500`: `lws_resolved` uses `max_length`, not actual context
The metadata sidecar computes `local_window_size_resolved` against the
configured `max_length` (7500), but each example may have been
middle-truncated to a shorter length, and the actual policy resolves
against the **per-example** prefill length. The recorded scalar therefore
doesn't reflect the policy actually used on those examples.

**Fix:** record `local_window_size_ratio` (the unresolved user-facing value)
in metadata, or compute the resolved value per example and store a
list/min/max summary. Either approach preserves run logic and clarifies the
sidecar.

### M4 — `longbench_runner.py:317-319`: middle truncation drops invariants
```python
prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + \
         tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
```
Decoding two halves and re-concatenating can re-tokenize to a *different*
token count than `max_length` (BPE merges at the seam, dropped specials,
etc). DefensiveKV's protocol middle-truncates **token IDs** directly.

**Fix:** keep IDs, don't round-trip through strings:
```python
input_ids = torch.cat([tokenized[:half], tokenized[-half:]])
# then run model.generate on input_ids directly (skip chat-template re-tokenize)
```
If the chat-template wrapper is required, splice template tokens around the
middle-truncated ID tensor rather than re-stringifying.

### M5 — `perf_runner.py:137,151,173`: `rope_module=rope or torch.nn.Identity()` silently masks RoPE absence
If RoPE discovery fails (no module with `.rotary_emb`), the cache is built
with `torch.nn.Identity`. `Identity` does not return `(cos, sin)`, so the
first eviction in `state.rerotate_keys` will raise. The fallback only
delays failure; it should fail fast.

**Fix:** raise `ConfigValidationError` from the discovery block when `rope
is None`, with a message pointing at the expected attribute path. Same
fix is appropriate in `ours_parity_runner.py:147-154` and
`longbench_runner.py:413-423` (currently they share the same
`or torch.nn.Identity()` pattern).

### M6 — `utils/config.validate_parity_pair:478`: missing fields silently pass
```python
if base_val is not None and ours_val is not None and base_val != ours_val:
```
If either side simply *omits* a field (rather than disagreeing), the check
is skipped. The list `_PARITY_IDENTITY_FIELDS` is meant to be an
identicality contract — a missing field on either side should fail loudly,
or at minimum log a warning.

**Fix:** convert the silent skip into a warning, or require all fields to
be present unless explicitly in a whitelist.

### M7 — `state.py:append`: aliases caller tensors on first call
```python
if self.key_states is None:
    self.key_states = key
    self.value_states = value
```
Later steps `torch.cat`, but the very first append stores the caller's
tensor by reference. If the caller mutates that tensor (or relies on it
being a separate buffer), the cache view changes silently.

**Fix:** `self.key_states = key.contiguous()` (or `.clone()`) on the
first-append branch. Negligible perf cost; eliminates an aliasing
landmine.

---

## Low severity (style / latent risk)

### L1 — `main.py:88` calls `_parse_value` defined at line 105
Works only because Python resolves names at call time. Move `_parse_value`
above `main()` (or make it a top-level helper imported into `main`) so
reordering doesn't surprise a future reader.

### L2 — `utils/seed.py:46`: global `torch.use_deterministic_algorithms(True)`
Some PyTorch ops (e.g. several scatter variants, `index_put_` w/
accumulate) have no deterministic implementation; toggling this globally
causes cryptic `RuntimeError`s in places unrelated to the experiment.
Consider scoping the deterministic flag to a context manager and only
enabling it inside runners that actually need bit-exact reproducibility.

### L3 — `utils/seed.py:39`: `os.environ["PYTHONHASHSEED"]` set after interpreter start
Setting it from inside the process has no effect on this process's hash
randomization (only on child processes). Drop the line or note this in
the docstring.

### L4 — `perf_runner.py:193`: throughput includes prefill time
```python
throughput_tokps = gen_len / max(gen_time + (t1-t0), 1e-9)
```
"Throughput (tok/s)" in TTFT/TPOT papers usually means decode-only
throughput. Mixing prefill into the denominator makes this number
incomparable to ChunkKV/H2O reports. Either rename the field
(`end_to_end_tokps`) or use only `gen_time` in the denominator.

### L5 — Cache implementations are duplicated verbatim
`modules/windowed_cache/cache.py` and `modules/windowed_eager_cache/cache.py`
are byte-identical (the backend split lives in `hooks.py`, not `cache.py`).
Any bug fix here must be applied twice — H5 is the immediate example.
Consider extracting `cache.py` into a shared module and importing it from
both packages.

### L6 — `modules/evaluation/faithfulness_runner.py:183`
```python
Sp_t = max(1, prefill_len + t - ns)
```
At step `t=0` the model has not yet emitted a new token, but the loop
treats the prefill as having grown by 0. The `W_act` cap is therefore one
window short on the very first iteration; usually invisible because that
first step rarely evicts.

**Fix:** if a strict off-by-one matters for analysis,
`Sp_t = max(1, prefill_len + t + 1 - ns)` to reflect the post-step sequence
length. Otherwise document the intent.

### L7 — `utils/hashing.sha256_tokenizer:63`
`sorted(vocab.items())` already produces a deterministic order; passing
`sort_keys=False` to `json.dumps` is a no-op there (the input is a list,
not a dict). Harmless, but a future maintainer may convert to `dict(...)`
and rely on `sort_keys`. Either keep the list form and drop `sort_keys`,
or convert to dict and set `sort_keys=True`.

### L8 — `longbench_runner.py:372`: newline-token detection
```python
newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
```
For BPE tokenizers a bare `"\n"` may tokenize differently from a `"\n"`
in context. The pick-last-token heuristic is fragile; spot-check the
chosen ID matches the model's actual newline behavior, or skip the
samsum-specific stop logic entirely if the dataset only needs a one-line
output (already handled by `_post_process`).

---

## How to approach fixes without changing logic

1. **Apply in the order H1 → H7 first.** Each is a localized edit (≤ 5
   lines) that corrects a stored value, a predicate, or a missing
   shape-pad branch. None of them changes scoring math, eviction policy,
   or cache layout.
2. **Add a regression test alongside each high-severity fix.** Suggested
   harness:
   - H1/H2: extend `modules/evaluation/test_base_parity.py` /
     `test_ours_parity.py` to assert the metadata field equals
     `policy.resolve(prefill_len, ...).local_tokens` (H1) and that
     `eviction_step_mask` aligns with the actual `cache.update` calls
     (H2).
   - H3/H4: a unit test in `tests/test_utils.py` that hands
     `jaccard_topk` arrays containing `-1` sentinels and asserts the
     score is unchanged from the no-sentinel baseline.
   - H5: drive `WindowedCache.update` with two consecutive
     `window_scores` where the second has fewer windows; assert no
     RuntimeError.
   - H6: use `torch.cuda.Event` (or a CPU monotonic check) to verify TTFT
     does not absorb pre-existing async work.
   - H7: parameterize `CacheConfig` with `local_window_size=True` and
     assert `ConfigValidationError`.
3. **For mediums, prefer the smallest predicate change** (e.g. `or` →
   `is None`, `or` → `and`). Avoid rewriting surrounding helpers.
4. **For the cache-duplication issue (L5)**, hold off on
   deduplication until H5 lands — it's easier to verify the fix on the
   two diverged copies first and then refactor once.
5. **Don't touch eviction math** (`policy.py`,
   `scorer.compute_window_scores`, `state.rerotate_keys`). The audit
   surfaced no bugs there; their behavior is load-bearing for parity
   with the saved base npzs.