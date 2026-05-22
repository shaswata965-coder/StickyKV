# StickyKV — Conversation Handoff

> Last updated: 2026-05-22  
> Branch: `main`  
> Purpose: Complete context for picking up this project in a new conversation without losing thread.

---

## 1. Project Overview

**StickyKV** is an H2O-style windowed KV-cache eviction system for LLaMA/Qwen models. Instead of keeping all past KV pairs, it divides the context into fixed-size windows and at each eviction step retains only the top-K highest-scored windows plus a local (most-recent) region and a fixed sink prefix.

**Goal of the evaluation suite**: Prove the eviction policy is faithful — i.e., that the windows it keeps are the same high-importance windows the full (unevicted) model would have prioritised.

---

## 2. Repository Layout

```
StickyKV/
├── main.py                          # top-level dispatcher (routes mode → runner)
├── modules/
│   ├── windowed_cache/              # flash-attn backend
│   │   ├── cache.py                 # WindowedCache — manages KV eviction
│   │   ├── state.py                 # CacheState — per-layer tensor storage
│   │   ├── policy.py                # EvictionPolicy — window-level retain logic
│   │   ├── config.py                # WindowedCacheConfig + ResolvedConfig
│   │   ├── scorer.py                # H2O cumulative score accumulation
│   │   ├── hooks.py                 # model hook installation
│   │   └── telemetry.py
│   ├── windowed_eager_cache/        # eager-attn backend (same API as above)
│   │   └── ...
│   └── evaluation/
│       ├── base_parity_runner.py    # Suite A base: vanilla DynamicCache + output_attentions
│       ├── ours_parity_runner.py    # Suite A ours: WindowedCache, teacher-forced
│       └── faithfulness_runner.py  # Suite B: post-processing only, no model load
├── scripts/
│   ├── kaggle_entry.py              # !python scripts/kaggle_entry.py --suite <name>
│   ├── print_faithfulness.py        # rich pretty-printer for faithfulness_results.npz
│   └── demo_generate.py
├── configs/
│   ├── eval_parity_base.yaml
│   ├── eval_parity_ours_eager.yaml
│   ├── eval_parity_ours_flash.yaml
│   ├── eval_faithfulness.yaml
│   └── ...
├── data/
│   └── corpus_loader.py             # CorpusLoader — wikitext-103 loader
└── utils/
    ├── config.py                    # ExperimentConfig + ParityValidationError
    ├── hashing.py
    ├── logger.py
    └── metrics.py                   # jaccard_topk, aggregate_*, heterogeneity
```

---

## 3. Core Concepts

### Cache Regions (per layer, per step)
```
[ sink tokens | evictable windows | local windows ]
   num_sink         top-K kept          always kept
                 (sorted by original position, not score)
```

- **window_size**: number of tokens per scoring window (e.g. 4)
- **num_sink_tokens**: always-retained prefix tokens (e.g. 4)
- **local_window_size**: most-recent N tokens always kept (e.g. 32 tokens = 8 windows of 4)
- **top_k_windows**: derived from `cache_budget` — how many evictable windows survive

### H2O Scoring
Cumulative attention mass per window. At every generation step, each query row contributes to key scores. Score is `sum(attn_weights)` across all query rows seen so far. No observation window — truly cumulative from step 0.

### original_window_ids
After eviction, compact window indices (0..W_compact-1) no longer correspond to original window positions. `CacheState.original_window_ids` is a tensor `[W_compact]` that maps each surviving compact window back to its **original** sequence position (0-indexed after sinks). This is essential for Jaccard alignment between base and ours — without it, the top-K indices from ours (compact space) cannot be compared to base (original space).

---

## 4. Evaluation Pipeline

### Step 1: Base Parity Runner
- Runs vanilla HuggingFace model with `DynamicCache` + `output_attentions=True`
- No eviction — full attention history kept
- At each generation step, records:
  - `window_scores [S, T, L, H, W]` — cumulative H2O scores (base's full context view)
  - `top_window_indices [S, T, L, K]` — top-K window indices **in original space**
  - `generated_tokens [S, T]` — token IDs for teacher forcing in ours runner
- Output: `parity_base_wikitext-103_<sha>.npz`

### Step 2: Ours Parity Runner
- Runs the same articles through WindowedCache (eager or flash backend)
- **Teacher-forced**: uses `generated_tokens` from base npz, never samples
- At each generation step, records:
  - `window_scores [S, T, L, H, W_compact]` — ours' compact-space scores
  - `top_window_indices [S, T, L, K]` — top-K indices translated to **original space** via `original_window_ids`
  - `retained_window_ids [S, T, L, M]` — ALL retained original window IDs (-1 padded); M = top-K + local windows, sorted by original sequence position
  - `retained_window_scores [S, T, L, H, M]` — ours' cumulative H2O scores for those retained windows (float16, -1 padded on M axis)
- Output: `parity_ours_eager_wikitext-103_<sha>.npz`

> **IMPORTANT**: Old ours npzs (pre-`0d88f5d`) do NOT contain `retained_window_ids` or `retained_window_scores`. If `FaithfulnessRunner` sees an old npz it raises `KeyError` with a clear message: "Re-run OursParityRunner to generate an updated npz."

### Step 3: Faithfulness Runner
- Pure post-processing, no model loaded
- Reads both npzs, validates metadata alignment (article sha, seed, prefill_len, gen_len, window_size, num_sink_tokens, model_name)
- For every `(step, layer)` computes 5 metrics over the **retained window set**:

| Metric | Formula | Ideal |
|--------|---------|-------|
| `cos_sim` | cosine similarity(ours_scores, base_scores) | → 1.0 |
| `pearson` | Pearson correlation | → 1.0 |
| `spearman` | Spearman rank correlation | → 1.0 |
| `kl_ours_base` | KL(ours ‖ base) | → 0.0 |
| `mass_ratio` | base_mass / ours_mass over retained windows | ≈ 1.0 |

- Also computes Jaccard (from top_window_indices, unchanged)
- Output: `faithfulness_results.npz` (schema v2.0)

### Step 4: print_faithfulness.py
Rich 7-section pretty-printer:
1. **Config header** — model, prefill, gen, window, budget, schema version
2. **Jaccard** — global + per-layer with prefill / Q1 / Q2 / Q3 / Q4 quartiles and drift arrow
3. **Master layer scorecard** — all 5 metrics (gen-mean) + Jaccard + composite score per layer, with █░ bar
4. **Generation trend** — for each of the 5 metrics: prefill + Q1/Q2/Q3/Q4 per layer, avg/std rows, drift
5. **Layer rankings** — best/worst 3 layers per metric using `L{n}(value)` format
6. **Head heterogeneity** — final-step Jaccard std across layers with █░ bar per layer
7. **Per-sample breakdown** — per-sample Jaccard global mean

**Composite score** = normalised average of: cos_sim, pearson, spearman, (1 - kl_inv), (1 - mr_inv), jaccard

---

## 5. Key Bugs Fixed and Why They Mattered

### Bug 1: Jaccard — window index misalignment (`dcff7b7`)
**Problem**: After eviction, ours' compact `window_scores` buffer shrinks. The top-K indices from ours were in compact space but compared to base indices in original space. Result: wrong Jaccard values.  
**Fix**: Added `original_window_ids` to `CacheState.__slots__`. Initialised as `torch.arange(W)` on first window assignment. Gathered alongside `window_scores` at every eviction. All recorded top-K indices for ours are now translated through `original_window_ids` before saving.

### Bug 2: Jaccard — K-mismatch padding (`dcff7b7`)
**Problem**: If base and ours had different K values (due to edge cases), Jaccard was compared on different-sized sets, inflating the intersection with -1 padding.  
**Fix**: Truncate both `base_tk` and `ours_tk` to `min(bK, oK)` before Jaccard computation.

### Bug 3: LIR local-window index offset (`89d5d05`)
**Problem**: LIR (Layer Information Retention — now removed) used zero-padded window_scores array but index math for local windows was wrong, pointing into the zero-pad region.  
**Fix**: Capped `W_act` to actual active windows using `math.ceil(Sp_t / ws_sz)`.

### Bug 4: LIR was using base scores on base-retained windows instead of ours-retained (`d2d4160`)
**Problem**: LIR was measuring what fraction of the base model's attention mass falls on the base's own top windows — which is trivially high. The comparison should be: for the windows ours retains, how similar are their score distributions.  
**Fix**: Changed to evaluate base scores on ours-retained windows. Later replaced entirely.

### Bug 5: LIR was replaced entirely with 5 distribution-comparison metrics (`0d88f5d`)
**Rationale**: LIR (~0.26) was actually *correct* — cache holds ~9% of windows, even with 2x H2O lift that's ~0.21–0.26. The number was not a bug but also not insightful. The real question is: "for the windows we do keep, how similar is our score vector to base's?" The 5 new metrics answer this directly. Requires storing ours' actual scores per retained window, hence the new npz arrays.

### Bug 6: Flash backend never produced window scores (`91ca8a7`)
**Problem**: `modules/windowed_cache/hooks.py` installed a monkey-patched attention `forward` that was an empty pass-through — it never set the `_captured_q` / `_captured_k` attributes the score hook read. The hook hit its early `return` on every call, so the flash backend produced no `window_scores` and eviction silently degraded to sink + local only (no H2O top-K selection).
**Fix**: Removed the broken monkey-patch. The score hook is now a plain `forward_hook` (registered `with_kwargs=True`) that recomputes the post-RoPE query from the layer's own inputs (`hidden_states` + `position_embeddings`) and reads keys from the cache. It also adds a causal mask to the auxiliary SDPA so prefill query rows do not attend to future keys — without it, the full N×N softmax inflated the scores of later windows.

> A 2026-05-22 cleanup pass (`2779ed1`) also removed confirmed dead code from `cache.py`, `policy.py`, `config.py`, and `metrics.py` (write-only attributes, uncalled methods, an unreachable branch, and the superseded LIR functions). No behaviour change — verified by the full CPU test suite.

---

## 6. NPZ Schemas

### parity_base npz (schema 1.1)
```
top_window_indices  [S, T, L, K]      int64   original-space top-K indices
window_scores       [S, T, L, H, W]   float16 cumulative H2O scores (full W, zero-padded)
eviction_step_mask  [S, T]            bool
generated_tokens    [S, T]            int64
metadata_json       [1]               object
```

### parity_ours npz (schema 1.1 + new arrays)
```
top_window_indices      [S, T, L, K]         int64   original-space top-K
window_scores           [S, T, L, H, W_c]    float16 compact-space scores
retained_window_ids     [S, T, L, M]         int64   ALL retained original IDs (-1 pad)
retained_window_scores  [S, T, L, H, M]      float16 ours scores for those windows (-1 pad)
eviction_step_mask      [S, T]               bool
metadata_json           [1]                  object
```

### faithfulness_results npz (schema 2.0)
```
jaccard           [T, L, 1]   Jaccard similarity (mean-over-heads)
jaccard_per_layer [T, L]
jaccard_global    [T]
heterogeneity     [L]         final-step Jaccard std across heads
cos_sim           [T, L]
pearson           [T, L]
spearman          [T, L]
kl_ours_base      [T, L]
mass_ratio        [T, L]
num_samples       [1]
per_sample_jaccard_global [S, T]
metadata_json     [1]
```

---

## 7. Kaggle Notebook Cell Order

The evaluation runs on Kaggle (GPU environment) because it needs the LLaMA model weights.

**Variables shared between cells** (set in earlier cells, read in later cells):
- `BASE_NPZ` — path to the base parity npz
- `OURS_NPZ` — path to the ours parity npz

### Cell 1 — Setup / installs
```python
!cd /kaggle/working && git clone https://github.com/shaswata965-coder/StickyKV.git
# or: !cd /kaggle/working/StickyKV && git pull origin main
```

### Cell 2 — Run Base Parity
```python
import os, sys, yaml, glob as _g
os.chdir("/kaggle/working/StickyKV")
if "/kaggle/working/StickyKV" not in sys.path:
    sys.path.insert(0, "/kaggle/working/StickyKV")

MODEL_PATH    = "/kaggle/input/models/metaresearch/llama-3.2/transformers/1b-instruct/1"
WIKITEXT_PATH = "/kaggle/input/datasets/shaswatabhattacharya/wiki-text-103-train/wiki.train.tokens"
PARITY_OUTPUT = "/kaggle/working/outputs/parity"

NUM_SAMPLES  = 3
MAX_TOKENS   = 1024
RATIO        = 0.5       # prefill=512, gen=512
CACHE_BUDGET = 0.20

from data.corpus_loader import CorpusLoader

def _local_load_wikitext103(self):
    with open(WIKITEXT_PATH, "r", encoding="utf-8") as fh:
        full_text = fh.read()
    articles = CorpusLoader._split_into_articles(full_text)
    return articles

CorpusLoader._load_wikitext103 = _local_load_wikitext103

base_cfg = {
    "run":   {"mode": "parity_base", "seed": 42},
    "model": {"name": MODEL_PATH, "dtype": "float16", "attn_implementation": "eager"},
    "cache": {"backend": "dynamic", "cache_budget": CACHE_BUDGET, "window_size": 4,
              "num_sink_tokens": 4, "local_window_size": 32},
    "data":  {"num_samples": NUM_SAMPLES, "max_tokens": MAX_TOKENS, "ratio": RATIO},
    "parity": {"dataset": "wikitext-103", "article_index": 0},
    "window": {"window_size": 4, "num_sink_tokens": 4, "local_window_size": 32},
    "telemetry": {"track_scores": False, "output_dir": PARITY_OUTPUT},
}

cfg_path = "/kaggle/working/parity_base_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(base_cfg, f, default_flow_style=False)

sys.argv = ["main.py", "--config", cfg_path]
from main import main as _main
_main()

npz_files = sorted(_g.glob(f"{PARITY_OUTPUT}/parity_base_wikitext-103_*.npz"))
BASE_NPZ = npz_files[-1]
print(f"Base NPZ → {BASE_NPZ}")
```

> **No changes needed** — BaseParityRunner was not modified.

### Cell 3 — Run Ours Parity (MUST re-run after any git pull)
```python
# Build ours config pointing at BASE_NPZ, then:
sys.argv = ["main.py", "--config", cfg_path]
from main import main as _main
_main()

ours_files = sorted(_g.glob(f"{PARITY_OUTPUT}/parity_ours_eager_wikitext-103_*.npz"))
OURS_NPZ = ours_files[-1]
print(f"Ours NPZ → {OURS_NPZ}")
```

> **Must be re-run** after pulling `main` post-`0d88f5d` because old ours npzs lack `retained_window_ids` / `retained_window_scores`.

### Cell 4 — Run Faithfulness
```python
import os, sys, yaml, glob as _g
os.chdir("/kaggle/working/StickyKV")
if "/kaggle/working/StickyKV" not in sys.path:
    sys.path.insert(0, "/kaggle/working/StickyKV")

PARITY_OUTPUT = "/kaggle/working/outputs/parity"
FAITH_OUTPUT  = "/kaggle/working/outputs/faithfulness"

base_files = sorted(_g.glob(f"{PARITY_OUTPUT}/parity_base_wikitext-103_*.npz"))
ours_files = sorted(_g.glob(f"{PARITY_OUTPUT}/parity_ours_eager_wikitext-103_*.npz"))
BASE_NPZ = base_files[-1]
OURS_NPZ = ours_files[-1]

faith_cfg = {
    "run": {"mode": "faithfulness", "seed": 42},
    "faithfulness": {"base_npz_path": BASE_NPZ, "ours_npz_path": OURS_NPZ},
    "telemetry":    {"output_dir": FAITH_OUTPUT},
}

cfg_path = "/kaggle/working/faithfulness_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(faith_cfg, f, default_flow_style=False)

sys.argv = ["main.py", "--config", cfg_path]
from main import main as _main
_main()
print(f"Faithfulness results → {FAITH_OUTPUT}/faithfulness_results.npz")
```

> **No changes needed** — cell is correct as-is.

### Cell 5 — Print Results
```python
exec(open("scripts/print_faithfulness.py").read())
# or: %run scripts/print_faithfulness.py
```

> Make sure `NPZ_PATH` at the top of `print_faithfulness.py` points to `{FAITH_OUTPUT}/faithfulness_results.npz`.

---

## 8. What "Faithful" Looks Like

For the 5 metrics, given the cache retains ~9-10% of windows with ~2x H2O lift:

| Metric | Expected range if working correctly |
|--------|-------------------------------------|
| cos_sim | > 0.7 (high — scores track base closely) |
| pearson | > 0.6 |
| spearman | > 0.6 (rank order preserved) |
| kl_ours_base | < 0.5 (low divergence) |
| mass_ratio | 0.8 – 1.2 (base and ours assign similar total mass to same windows) |
| jaccard | ~0.60–0.75 (previously measured ~0.66–0.71) |

**Why LIR was ~0.26 (and why that's correct, not a bug)**:
- Cache budget 20% of prefill → top_k ≈ 24 windows
- Base accumulates 127–255 total windows over generation
- 24 / 255 ≈ 9% coverage
- H2O lift ≈ 2.3x over uniform baseline (uniform = 9%, LIR ≈ 9% × 2.3 ≈ 0.21–0.26)
- LIR of ~0.26 with a 2x H2O lift is geometrically expected. It was not a bug.
- Replaced by the 5 distribution-comparison metrics which are more informative.

---

## 9. Pending Work

| Task | Status | Notes |
|------|--------|-------|
| Re-run OursParityRunner on Kaggle | **Required next** | Any existing ours npz predates `retained_window_ids` / `retained_window_scores` |
| Re-run FaithfulnessRunner | Blocked on above | Will work immediately after new ours npz is available |
| Read and interpret the 5-metric output | Next analysis step | No code changes needed |
| Validate flash-attn backend end to end | Hooks fixed (`91ca8a7`) | Score capture + eviction now work; needs a real GPU + flash-attn run to confirm |
| LongBench evaluation | Not started | `longbench_runner.py` exists, not recently touched |

### Known issues (2026-05-22 code review — not yet fixed)

| Issue | Where | Notes |
|-------|-------|-------|
| Windowed perf configs never evict | `perf_runner.py` `_measure_config` | Score hooks are installed against a throwaway cache; the live forwards use different cache objects, so `window_scores` never reaches them — windowed perf numbers measure a non-evicting cache. |
| `eviction_step_mask` off by one | `ours_parity_runner.py:213` | Reads `cache._generation_step` after `update()` already incremented it — flags step r when eviction happened at r+1. Currently latent (no consumer). |
| Stale visualization schema | `visualize.py` | `make_lir_trajectory` / `make_missed_mass_distribution` / `make_budget_sweep` read `global_lir` arrays the faithfulness runner stopped emitting after `0d88f5d`; those 3 plots render empty. |
| transformers version mismatch | `environment.yml` vs installed | `environment.yml` pins `transformers>=4.36,<4.46`; on transformers 5.x `WindowedCache` crashes in `create_causal_mask` (`get_mask_sizes` / missing `.layers`). Run with a 4.36–4.45 transformers. |

---

## 10. Git Commit History (recent, most relevant)

| Hash | Message |
|------|---------|
| `2779ed1` | Remove dead code from cache, policy, config, and metrics modules |
| `91ca8a7` | Fix flash-attn backend score hooks: implement query capture, add causal mask |
| `a1622e6` | Add three deep-dive documentation files for StickyKV |
| `080b85f` | Add Handoff.md: comprehensive session context for new conversations |
| `cb4b490` | Enrich print_faithfulness: master scorecard, generation quartile trends, layer rankings |
| `0d88f5d` | Replace LIR with 5 distribution-comparison metrics per (step, layer) |
| `449cf3d` | Downgrade LIR diagnostics from INFO to DEBUG |
| `7b1ae74` | Merge PR #7 feat/lir-diagnostic-logging |
| `57af19b` | Add LIR diagnostic logging at representative steps |
| `89d5d05` | Fix LIR local-window index offset into zero-padding |
| `d2d4160` | Fix LIR proxy: use base attention mass on ours-retained windows |
| `dcff7b7` | Fix Jaccard similarity bugs: window index misalignment and K-mismatch padding |

---

## 11. Design Decisions and Rationale

**Why teacher-forced evaluation?**  
If ours runner sampled its own tokens, generation would diverge from base after the first eviction. Teacher forcing ensures both runners see the same input at every step, making the comparison meaningful.

**Why window-granularity scoring (not token-granularity)?**  
Eviction operates at window granularity (groups of `window_size` tokens). Comparing windows is the correct unit; token-level comparison would be finer than what the policy actually controls.

**Why sort retained windows by original position?**  
Top-K selected by score descending; local windows appended. Both need to be in sequence order for score vectors from ours and base to be position-aligned when computing cos/Pearson/Spearman/KL/mass-ratio.

**Why Option B (store ours' scores) over Option A (re-run base on ours' windows)?**  
Option A would require a second base forward pass. Option B stores the scores that ours already computed during its normal forward pass — zero extra compute, just extra storage in the npz.

**Why replace LIR entirely rather than fix it?**  
LIR answers "how much of base's attention mass does our cache capture?" The answer (~0.26) is correct and theoretically expected for a 9% cache with 2x lift. The more useful question is "given the windows we keep, do our scores agree with base's scores?" That's what the 5 new metrics answer.

**Why `mass_ratio = base_mass / ours_mass` rather than the reverse?**  
Convention: numerator is the reference (base). If ratio > 1, base assigns more weight to these windows than ours does. If ≈ 1, they agree. Ours_mass in denominator is clamped to `1e-8` to avoid division by zero.
