# StickyKV Quantization — Rationale & Open Items

Supporting analysis and trade-off studies behind the **fixed** choices in
[design.md](design.md). Nothing here is ratified spec — it is the "why", the cost
comparisons, the options still under evaluation, the integration audit, and the items
left to decide. Once an item is settled it graduates to design.md; once an alternative
is rejected outright it moves to [design_history.md](design_history.md).

Target precision is **int4** (the int8 milestone was dropped).

---

## Outlier strategy (int4)

The int4 outlier choice is **not yet ratified**. This section records the analysis.

### Why the burden is already light

Three properties mean StickyKV needs far less than a uniform-quant cache (KVQuant
keeps ~1% of entries in fp16; that is overkill here):

1. **Per-channel keys** give each channel its own scale, neutralising the dominant
   inter-channel key-outlier structure.
2. **Pre-RoPE storage** keeps that per-channel structure consistent (RoPE smears it),
   so per-channel scales stay tight.
3. **Two-tier split** puts the highest-attention windows in fp16; the Q tier holds
   only mid-importance windows, so its worst outliers are disproportionately *already*
   in the fp tier.

### The three option families

| option | mechanism | wrinkle in this system |
|---|---|---|
| **A. Dense-and-sparse** (KVQuant) | keep top ~1% entries in an fp16 side-list | irregular gather/scatter breaks the clean layout; best quality |
| **B. Hadamard rotation** (QuaRot/RotateKV) | rotate channels to spread outliers | conflicts with per-channel keys; **does not commute with read-time RoPE on keys** — but is free on *values* |
| **C. Non-uniform codebook** (NF4/NUQ) | distribution-matched 16-level LUT | cost is phase-dependent — see below |

### Is NF4 costly? — it depends on the phase (this corrects an earlier over-statement)

The earlier draft deferred NF4 by charging it the *Phase 2 tile* cost. That was wrong
for v1. Split by phase:

- **Phase 1 (v1, materialize):** NF4 dequant is `codebook[codes] * scale` — a single
  vectorized gather (`torch.take`/`F.embedding`) over the Q tier, memory-bound, the
  same order as the affine multiply. **Effectively free in v1.** So there is no v1 cost
  reason to withhold it — this is the answer to "why not do it right away": *for v1, do
  it.*
- **Phase 2 (fused Triton tile):** here it matters. Affine dequant is one FMA (pure
  ALU, hidden under memory latency — the property that makes the fused tile free). NF4
  replaces that FMA with a 16-entry LUT lookup (register shuffle, or shared-mem load
  with possible **bank conflicts**) that can *serialize*. So Phase 2 revisits whether
  to keep NF4 in-tile or fall back to affine keys for the kernel — a kernel decision,
  not a reason to withhold NF4 from v1.
- **Calibration:** a fixed NF4 codebook (QLoRA constants) has zero calibration cost; a
  per-channel NUQ codebook (k-means) adds offline work per demotion. Use the fixed
  codebook to keep the pinned-grid freeze trivial.

### Recommended int4 approach (revised)

Adopt both cheap pieces up front; keep only the genuinely-costly piece contingent:

- **Values → Hadamard rotation folded into the weights (free, always-on).** Values
  carry **no RoPE**, so a shared Hadamard `H` on head_dim folds entirely offline: `H`
  into `W_v` (values born rotated, `V·H = x·(W_v·H)`) and `Hᵀ` into `W_o` (output
  un-rotates, `(P·V·H)·(Hᵀ W_o) = P·V·W_o`). Zero extra decode FLOPs, zero extra
  storage, lossless on the fp tier. Adopt.
- **Keys → NF4-style non-uniform codebook, per-channel, no rotation (cheap in v1).**
  Keep the pinned per-channel absmax scale; swap uniform levels for a fixed 16-entry
  codebook. Free in the v1 materialize path (above); revisit for the Phase 2 tile.
- **Contingency (the one with real operational cost even in v1): micro dense-sparse
  (~0.1–0.25%).** An fp16 side-list needs an irregular gather and breaks the clean
  chronological layout, so add it **only if** the int4 LongBench gate misses — kept far
  smaller than KVQuant's ~1% because the fp tier already holds the biggest spikes.

So the split is now by *actual cost*, not caution: value-fold and NF4 keys are the
baseline (both cheap in v1); dense-sparse is the sole contingency.

### Cost: recommended baseline vs no outlier handling (v1 / materialize)

| axis | no outlier | value Hadamard fold | NF4 keys | (contingency) micro-sparse |
|---|---|---|---|---|
| **v1 decode compute** | baseline | **+0** (offline fold) | ~0 (vectorized gather) | +irregular gather (~0.1–0.25%) |
| **storage** | baseline | **+0** | ≈0 (may drop the zero-point) | +~0.1–0.25% fp16 |
| **offline prep** | — | fold `H` into `W_v`,`W_o` once | pick 16-level codebook | select outliers per demotion |
| **Phase 2 tile** | 1 FMA | +0 | LUT lookup (may serialize) | breaks tile locality |

**Bottom line:** the recommended int4 baseline costs ≈ no-outlier in v1 (both additions
are free/near-free in the materialize path). Real cost appears only in the Phase 2 tile
(NF4 LUT) and in the optional sparse net. And the dominant v1 cost overall is the fp16
write-back of the materialized Q tier (design.md §8) — outlier handling is second-order.

---

## Integration audit — required changes to current StickyKV code

Grounded against the current `windowed_cache` (mirrored in `windowed_eager_cache`).
Roughly ordered from cache-core outward.

1. **`policy.py` — two-tier split.** `compute_retain_window_indices` does one
   `torch.topk` over evictable windows. Needs: rank → top `N_fp` (incl. sink+local) to
   fp, next `N_q` to Q, rest dropped; return both retained sets.
2. **`config.py` — resolver.** Add `quant_ratio q`, bit-width, scale dtype; compute
   `N_fp`, `N_q`, `b_q`. `ResolvedConfig` gains fields (design.md §7).
3. **`state.py` — Q store + ledger.** New packed-int4 store (channel-major keys,
   token-major values) + per-window ledger (`original_window_id`, `codes`, `scale`,
   `zero`, `offset`, `position_range`).
4. **`state.py` — `rerotate_keys(new_pos)`.** Add the explicit interleaved
   target-position argument; fix the trailing `position_ids` write to the gappy fp
   positions, not `arange(T_retained)` (design.md §5).
5. **New `build_interleaved_position_map`** + wire into the `cache.py` eviction path.
6. **`cache.py` — `update()` eviction.** Insert tier assignment + demote/promote data
   movement + interleaved map + ledger update. **Return value changes**: from the live
   `state.key_states/value_states` to `materialize_effective_kv(...)` — a freshly-built
   chronological interleave, not the raw fp cache. (Understated before as an "order-free
   concat".)
7. **New `materialize_effective_kv`** (shared) — dequant Q + RoPE at `position_range` +
   chronological interleave with fp by `original_window_id`.
8. **`cache.py` — `get_seq_length` returns effective `T_total`** (fp + Q), since HF uses
   it for mask sizing and `position_override`'s `past_seen`. Reporting fp-only length
   desyncs the query position from the effective key count.
9. **`position_override.py` — `past_seen = T_total`.** Follows from (8); verify Q tokens
   are counted (design.md §5).
10. **`hooks.py` (flash) — score hook** reads `cache._states[l].key_states` directly
    (line ~219); must instead source effective K via `materialize_effective_kv`, or it
    scores only the fp tier and misses the interleave. **The one required flash-hook
    change.** (Eager attends over `update()`'s return, so its real-attention scoring
    needs no hook change — but it *does* depend on (6)/(7)/(8).)
11. **Value Hadamard fold (if adopted) is model surgery, outside the cache.** Folding
    `H`→`W_v` and `Hᵀ`→`W_o` is a model-load step (e.g. in `cache_factory`/model setup);
    no weight-modification step exists today. Larger integration surface than the
    cache-local key changes — worth scoping explicitly.
12. **Mirror to `windowed_eager_cache`.** All `cache.py`/`state.py` edits stay
    byte-identical; the quant module + ledger are shared.

Already anticipated by design.md: (4),(5),(9) via §5; (7) via §9; (1)-(3) via §5–§7.
Newly surfaced / previously understated: (6) return-value + chronological interleave,
(8) effective length, (10) flash-hook read, (11) value-fold as model surgery.

---

## Open items (resolve before the affected milestone)

- **Scale dtype (blocks a deterministic `b_q`).** fp16 / fp8 for the pinned scale/zero.
  Swept in Suite C; the resolver needs a default to compute `N_q`.
- **Quantizer numerics.** Pin the exact asymmetric affine quant/dequant formula and the
  NF4 codebook (fixed QLoRA constants vs per-channel percentiles) against a KIVI/NF4
  reference before coding. Nibble packing is already decided (design.md §2).
- **int4 sparse-net gate.** Decide the LongBench threshold that triggers the micro
  dense-sparse escalation.
- **Explicit hysteresis.** Deferred; revisit only if Suite A Jaccard shows measurable
  K/Q boundary churn.
