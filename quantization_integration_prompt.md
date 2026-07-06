/# Implementation Prompt — Integrate the StickyKV Quantization Design

You are implementing the two-tier windowed KV cache described in [design.md](design.md)
into the existing StickyKV codebase. This prompt translates that design into a
concrete, ordered work plan bound to the files that already exist. Read
[design.md](design.md) in full first — it is the spec. This document tells you
_where_ each piece lands and _which invariants must not break_.

Supporting rationale is in [design_rationale.md](design_rationale.md); the decision
record is in [design_history.md](design_history.md). Do not re-open decisions those
files mark as fixed.

---

## 0. Ground rules (violating any of these is a regression)

1. **`p=1` / `q=0` byte-identical.** With the quant tier disabled (`quant_ratio q = 0`,
   or the feature flag off) the cache must behave **bit-identically** to today.
   Every new code path is gated so the pure-fp16 path is untouched. Confirm with the
   existing `modules/evaluation/test_*_parity.py` suites.
2. **Backend mirroring.** `modules/windowed_cache/{cache.py,state.py}` and
   `modules/windowed_eager_cache/{cache.py,state.py}` are **byte-identical** twins
   (see the header banners in those files). Any edit to one is mirrored verbatim to
   the other. The **only** legitimate divergence stays in `hooks.py` (§9 of the
   design). The new quantizer/ledger/store is a **single shared module imported by
   both backends** — do not duplicate it per-backend.
3. **B=1 only in v1.** The read path relies on `utils/position_override.py` (a B=1
   construct) and the interleaved map runs **per row**. Keep every new primitive
   per-row so B>1 is a later extension, not a rewrite. Do not add a B>1 code path.
4. **CPU-testable / transformers pin.** The dev box runs transformers 5.8.1, but the
   target is 4.47.1 and `utils/cache_factory.py` refuses to run model-backed on
   > 4.47.1. So **all new logic must be exercised by `pytest -m "not gpu"` CPU unit
   > tests** — no full-model forward is required to prove correctness. Design them so
   > they run with a tiny fake rope module and hand-built tensors, as the current
   > `tests/test_windowed_cache.py` does.
5. **No re-quantization, ever (§3, §10).** Codes are written exactly once per window
   lifetime. Re-demotion is a _reactivation_ of the dormant ledger entry, never a
   recompute. There must be no arithmetic path that runs `quant(dequant(·))`.
6. **Pinned grid stored in fp16 (§2).** Quantize against the **fp16-rounded**
   `scale`/`zero`, not the fp32 intermediates, so the grid the codes were fit to is
   bit-identical to the grid used at dequant.

---

## 1. Current-code map (what you are extending)

| Concern                | File(s)                                          | Relevant today                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF Cache orchestration | `modules/windowed_cache/cache.py` (+ eager twin) | `update()` appends, accumulates scores, evicts (rank→retain window idx→token idx→`slice_and_keep`→`rerotate_keys`→gather scores/ids), returns live `state.key_states/value_states` |
| Tensor storage         | `modules/windowed_cache/state.py` (+ twin)       | `key_states/value_states/position_ids/window_scores/original_window_ids`; `append`, `slice_and_keep`, `rerotate_keys(rope, old_pos)` (rebases to `arange(T_retained)`)             |
| Eviction index math    | `modules/windowed_cache/policy.py` (+ twin)      | `compute_retain_window_indices`, `expand_to_token_indices` (arithmetic `num_sink + w*window_size + offset`)                                                                        |
| Scoring                | `modules/windowed_cache/scorer.py` (+ twin)      | `compute_window_scores`, `reduce_token_scores_to_windows`, `accumulate` — Lp power-sums on the merged window axis                                                                  |
| Budget resolver        | `modules/windowed_cache/config.py` (+ twin)      | `WindowedCacheConfig` / `ResolvedConfig`; `resolve()` computes `bytes_per_token`, `total_budget_bytes`, `top_k_windows`                                                            |
| Flash score hook       | `modules/windowed_cache/hooks.py`                | reads **raw** `cache._states[lidx].key_states` for the aux SDPA                                                                                                                    |
| Eager score hook       | `modules/windowed_eager_cache/hooks.py`          | reads `attn_weights` from the module output — i.e. attends over `update()`'s return                                                                                                |
| Query position         | `utils/position_override.py`                     | sets query at `cache.get_seq_length(0)`                                                                                                                                            |
| Backend/version gate   | `utils/cache_factory.py`                         | `get_cache_classes`, `assert_transformers_version_supported`                                                                                                                       |

---

## 2. New shared module to create

Create a **single shared package** `modules/quant/` (imported by both backends):

- `quantizer.py` — the hand-rolled KIVI-style affine int4 quantizer. Implement
  **exactly** the numerics in design §2:
  - `scale = (mx − mn)/15`, `zero = mn` (float offset), computed in fp32, then
    **stored/rounded to fp16** and re-loaded before the codes are fit.
  - `q = clamp(round_half_even((x − zero)/scale), 0, 15)`, clamp **before** the uint
    cast; `x̂ = q·scale + zero`.
  - Degenerate `mx == mn` → `scale = 1`, all codes 0, `x̂ = mn` exactly.
  - Key granularity: per-`(head, channel, window)` (channel-major store
    `[H_kv, D, T_q]`, pack 2 tokens/byte → `[H_kv, D, ceil(window/2)]`; scale/zero
    `[H_kv, D]` per window). Value granularity: per-token (token-major
    `[H_kv, T_q, D]`, pack 2 channels/byte; scale/zero `[H_kv, T_q]`).
  - **Bring-up path:** implement an _unpacked_ codes variant first (one 4-bit value
    per uint8 so `torch.gather` works token-wise), get all round-trip/parity tests
    green, then switch to the nibble-packed layout behind the same API. `window_size`
    even is required (assert it) so there is no tail padding.
- `store.py` — `QuantizedStore`: gap-free dense int4 code store + per-window fp16
  scales/zeros + `position_ids`, keys stored **pre-RoPE**. Supports append, compact
  (offset shift), and dequant-at-read.
- `ledger.py` — the per-window ledger keyed by `original_window_id` (design §6 table):
  `original_window_id` (immutable), `codes` (immutable), `scale`/`zero` (immutable,
  pinned), `offset` (mutable), `position_range` (mutable). Entries persist through
  promotion as **dormant** (codes+grid retained, `offset` invalid, excluded from
  reads/interleave); freed only on outright drop. Re-demotion = reactivate.

Everything in `modules/quant/` must be pure-tensor and CPU-testable with no
transformers dependency beyond the rope module passed in at read time.

---

## 3. Ordered workstreams

Do these in order; each ends green before the next starts. Keep the quant tier behind
a config flag / `q=0` default so `main` stays shippable throughout.

### WS-1 — Quantizer + store + ledger (pure, no cache wiring)

- Implement `modules/quant/{quantizer,store,ledger}.py` per §2 above.
- CPU tests: round-trip error bounds; degenerate-group exactness (`mx==mn` → `x̂=mn`);
  fp16-grid idempotence (`quant` against fp16 grid, dequant, re-quant against the same
  grid → **bit-identical codes**); pack/unpack round-trip equals the unpacked variant.
- No changes to `cache.py`/`state.py` yet.

### WS-2 — Tier-aware budget resolver (design §7)

- Extend `WindowedCacheConfig`/`ResolvedConfig.resolve()` in `config.py` (mirror to
  twin) with new knobs: `quant_ratio` (`q`, default **0.0** = feature off), bit-width
  (fix 4 in v1), group size (= `window_size`). Scale dtype is **fixed fp16 — not a
  knob** (§2, §7).
- Compute `M_budget = β·M_full`, `M_fp=(1−q)·M_budget`, `M_q=q·M_budget`,
  `N_fp = M_fp/b_fp`, `N_q = M_q/b_q` with `b_fp = 4·H_kv·D·window_size` and the
  `b_q` formula in §7 (packed int4 K+V **plus** the fp16 key and value scale/zero
  overhead). **Use `b_q`, not `b_fp`, for the Q tier.** Sink+local stay inside `M_fp`:
  `top_k_fp = N_fp − (sink + local)`.
- Preserve the existing `bytes_per_token/total_budget_bytes/top_k_windows` outputs so
  `q=0` reproduces today's `ResolvedConfig` exactly (byte-identical parity).
- CPU tests: the §7 worked example (β=0.25, q=0.5 → N_q≈4·N_fp); `q=0` equals the
  current resolver output field-for-field.

### WS-3 — Merged window axis + interleaved position map (design §5)

This is the structural core. Add to `state.py`/`policy.py` (mirror to twins) and
`modules/quant`:

- `build_interleaved_position_map(fp_window_ids, q_window_ids, window_size, num_sink)`
  — **per row**; sort all surviving window ids jointly by `original_window_id`, assign
  one contiguous `arange(T_total)` (`T_total=T_fp+T_q`), return **(a)** fp-survivor
  target positions for `rerotate_keys` (possibly gappy) and **(b)** per-Q-window
  `position_range` for the ledger.
- `rerotate_keys(rope, old_pos, new_pos)` — add the explicit `new_pos` argument
  (currently hard-codes `arange(T_retained)`). **Critical:** its trailing
  `self.position_ids` bookkeeping must be set to the **interleaved (gappy) fp
  positions**, not `arange(T_retained)`, or the next eviction snapshots wrong "old"
  angles. Keep the old behaviour as the default when `new_pos is None` for `q=0`
  parity.
- Make `expand_to_token_indices` **tier-aware**: expand only the **fp partition** of
  the retained merged indices to fp-store token indices (fp-store window rank =
  cumsum over the fp-tier mask → `num_sink + rank_fp·window_size + offset`); Q windows
  resolve through the ledger and get **no** token gather.
- CPU tests: interleaved-map correctness (fp slots skip over Q slots; gaps land where
  Q windows sit); per-row divergent eviction; `rerotate_keys(new_pos)` leaves
  `position_ids` = the gappy fp positions.

### WS-4 — `materialize_effective_kv` + two-tier `update()`/eviction (design §5, §8, §9)

- Add `materialize_effective_kv(fp_store, q_store, ledger, rope)` (shared): dequantize
  the Q store, apply RoPE at each Q window's current `position_range`, interleave with
  the fp store **by `original_window_id`** into chronological order (a gather over
  `N_fp+N_q` windows, **not** a `[fp ‖ Q]` concat). Returns effective K/V.
- Rewire the eviction cycle in `cache.py` (mirror to twin) to the 7 steps in §5:
  rank on the merged axis → assign tiers (`N_fp`/`N_q` from WS-2) → move
  boundary-crossers (demote: un-rotate once via the `rerotate_keys` un-rotate half,
  quantize against a freshly-pinned fp16 grid, append to Q store, drop fp copy;
  **re-demote = reactivate dormant ledger entry, no re-quant**; promote: dequantize,
  append to fp store, keep ledger entry **dormant**) → `build_interleaved_position_map`
  → `rerotate_keys` fp tier to its interleaved slots → update ledger `position_range`
  (O(Q_windows) int writes, no tensor move, no re-quant) → `position_override` uses
  `T_total`.
- `update()` returns `materialize_effective_kv(...)` (a freshly-built interleaved
  tensor) instead of the live `state.key_states/value_states`. Callers must not assume
  the return aliases the stored fp cache — document this at the return site.
- `cache.get_seq_length()` must report the **effective** `T_total` (fp+Q), since HF
  and `position_override` consume it for mask sizing / `past_seen`. Reporting fp-only
  length desyncs the query position.
- Gate the whole thing on `q>0`; `q=0` takes the legacy single-tier path unchanged.
- CPU tests: promote→demote reactivation (codes **bit-identical**, no recompute);
  position-invariance of attention under interleave order; merged-axis score alignment
  (`window_scores` index ↔ chronological window id across a **mixed-tier** eviction).

### WS-5 — Flash hook sources effective K (design §8, §9)

- The **one required flash-hook change**: `modules/windowed_cache/hooks.py` currently
  reads raw `cache._states[lidx].key_states` for the aux SDPA. With a live Q tier this
  misses the Q windows and the interleaving. Source the effective K via
  `materialize_effective_kv` instead so scoring covers both tiers on the merged axis.
- The eager hook already attends over `update()`'s return, so it needs **no** change
  beyond confirming that return is now the effective K/V. Do **not** add an aux SDPA
  to eager.
- CPU test: flash/eager parity of `window_scores` across a mixed-tier eviction.

### WS-6 — Config surface + gates

- Expose `quant_ratio` (and bit-width/group-size if you surface them) in the YAML
  config (`utils/config.py`) and the LongBench runner, matching how `score_p` was
  threaded through recently. Default `q=0`.
- Wire telemetry: promotion-frequency counters (design §10) and dormant-entry counts.
- Do **not** change `MAX_SUPPORTED_TRANSFORMERS`; the mask-API gap on 5.x is separate.

---

## 4. Test matrix (all `pytest -m "not gpu"`, CPU)

Mirror the style of `tests/test_windowed_cache.py` and
`modules/evaluation/test_*_parity.py`. Required cases (design "CPU unit tests" list):

- Quantizer round-trip error within bound; degenerate-group exactness; fp16-grid
  idempotence; pack/unpack equals unpacked.
- Promote→demote **reactivation**: codes bit-identical, zero recompute calls
  (assert the quantizer is not re-invoked, e.g. via a call counter/spy).
- Position-invariance: attention output equal regardless of physical interleave order.
- Interleaved-map correctness: fp gaps land exactly over Q slots; per-row divergence.
- Merged-axis score alignment across a mixed-tier eviction.
- Flash/eager parity on the merged axis.
- **Regression parity:** with `q=0`, `resolve()` output and full `update()`/eviction
  behaviour are byte-identical to `main` (run the existing parity suites).

---

## 5. Deliverable shape

- New `modules/quant/` package (shared, single copy).
- Edits mirrored byte-for-byte across `windowed_cache` and `windowed_eager_cache`
  for `cache.py`/`state.py`/`policy.py`/`config.py`/`scorer.py`; **only `hooks.py`
  diverges** (flash gets the `materialize_effective_kv` source change; eager unchanged).
- `utils/position_override.py`: `T_total` positioning follows automatically from
  `get_seq_length()` returning `T_total` — verify, don't hard-code a second path.
- All new behaviour gated so `q=0` is the untouched legacy path.
- Phase 1 only (materialize-then-interleave, §11). **Do not** build the Triton
  (Phase 2) or FlashInfer (Phase 3) kernels — but keep the store layout
  (tile=window=scale-group, pre-RoPE codes) so Phase 2 is a drop-in later.

Work WS-by-WS, keeping the suite green and `q=0` byte-identical at every step.
Report which design section each commit implements.
