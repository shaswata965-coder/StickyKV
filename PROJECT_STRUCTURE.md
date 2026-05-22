# Project Structure — End-to-End Overview

This document explains the purpose of every folder, where each script lives,
what the bash scripts do, and how all the pieces fit together.

---

## 1. Directory Tree

```
C:\StickyKV/
├── main.py                          # Single CLI entry point for all modes
│
├── configs/                         # YAML experiment configurations
│   ├── base.yaml                    # Shared defaults inherited by all others
│   ├── eval_parity_base.yaml        # Suite A base (full cache, DynamicCache)
│   ├── eval_parity_ours_eager.yaml  # Suite A ours (eager backend)
│   ├── eval_parity_ours_flash.yaml  # Suite A ours (flash-attn2 backend)
│   ├── eval_faithfulness.yaml       # Suite B (score distribution comparison)
│   ├── eval_perf.yaml               # Suite C (latency/throughput benchmarks)
│   ├── eval_visualize.yaml          # Visualization runner
│   ├── longbench_full_cache.yaml    # Suite D baseline (full DynamicCache)
│   ├── longbench_ours_eager.yaml    # Suite D ours (eager backend)
│   └── longbench_ours_flash_attn.yaml  # Suite D ours (flash-attn2 backend)
│
├── data/                            # Corpus loading and article identity
│   ├── corpus_loader.py             # Loads wikitext-103 / PG19 from HuggingFace
│   └── article_registry.py         # SHA-based article identity tracking
│
├── modules/                         # Core implementation modules
│   ├── windowed_cache/              # Flash-attention backend (default)
│   │   ├── cache.py                 # WindowedCache — HF Cache integration, orchestrates eviction
│   │   ├── policy.py                # EvictionPolicy — top-K window selection, trigger logic
│   │   ├── state.py                 # CacheState — per-layer KV tensors + scores + positions
│   │   ├── scorer.py                # compute_window_scores() + accumulate()
│   │   ├── hooks.py                 # install_score_hooks() — forward hook + auxiliary SDPA
│   │   ├── config.py                # WindowedCacheConfig, ResolvedConfig
│   │   └── telemetry.py             # Telemetry / NullTelemetry for score recording
│   │
│   ├── windowed_eager_cache/        # Eager-attention backend (Kaggle / no flash-attn)
│   │   ├── cache.py                 # Identical to flash backend
│   │   ├── policy.py                # Identical to flash backend
│   │   ├── state.py                 # Identical to flash backend
│   │   ├── scorer.py                # Identical to flash backend
│   │   ├── hooks.py                 # Different: reads materialized attn_weights
│   │   └── config.py                # Identical to flash backend
│   │
│   └── evaluation/                  # All four evaluation suite runners
│       ├── base_parity_runner.py    # Suite A — baseline (full cache reference run)
│       ├── ours_parity_runner.py    # Suite A — ours (windowed cache, teacher-forced)
│       ├── faithfulness_runner.py   # Suite B — score distribution comparison (no model)
│       ├── perf_runner.py           # Suite C — latency / throughput benchmarks
│       ├── longbench_runner.py      # Suite D — LongBench generation
│       ├── longbench_scoring.py     # Suite D — post-hoc metric scoring
│       ├── longbench_metrics.py     # Suite D — vendored THUDM metrics (do not modify)
│       └── visualize.py             # Visualization runner
│
├── scripts/                         # Bash scripts for running evaluation suites
│   ├── reproduce_all.sh             # Master script: runs all suites end-to-end
│   ├── run_parity_base.sh           # Suite A base
│   ├── run_parity_ours_eager.sh     # Suite A ours (eager)
│   ├── run_parity_ours_flash.sh     # Suite A ours (flash-attn2)
│   ├── run_faithfulness.sh          # Suite B
│   ├── run_perf.sh                  # Suite C
│   ├── run_visualize.sh             # Visualization
│   ├── run_longbench_full_cache.sh  # Suite D baseline
│   ├── run_longbench_ours_eager.sh  # Suite D ours (eager)
│   ├── run_longbench_ours_flash_attn.sh  # Suite D ours (flash-attn2)
│   └── score_longbench.sh           # Suite D scoring (no model, post-hoc)
│
├── tests/                           # Unit tests
│   └── *.py                         # Per-module unit tests
│
└── utils/                           # Shared utilities
    ├── config.py                    # Typed config dataclasses + load_config() + validate_parity_pair()
    ├── cache_factory.py             # get_cache_classes() — backend selection + pairing validation
    ├── metrics.py                   # Jaccard similarity + aggregation helpers
    ├── hashing.py                   # sha256_file(), sha256_text()
    ├── logger.py                    # get_logger() — structured logging setup
    └── seed.py                      # seed_everything() — Python + NumPy + PyTorch seeding
```

---

## 2. The Single Entry Point — `main.py`

All modes run through one CLI:

```bash
python main.py --config configs/<name>.yaml [--override key=value ...]
```

`main.py` does three things:
1. Parses `--config` and optional dot-notation `--override` arguments.
2. Calls `load_config()` from `utils/config.py` to produce a typed `ExperimentConfig`.
3. Looks up the runner class in `_RUNNER_REGISTRY` and calls `runner.run()`.

```python
# main.py:25-33  — Mode → Runner mapping
_RUNNER_REGISTRY = {
    "parity_base"    : "modules.evaluation.base_parity_runner.BaseParityRunner",
    "parity_ours"    : "modules.evaluation.ours_parity_runner.OursParityRunner",
    "faithfulness"   : "modules.evaluation.faithfulness_runner.FaithfulnessRunner",
    "perf"           : "modules.evaluation.perf_runner.PerfRunner",
    "longbench"      : "modules.evaluation.longbench_runner.LongBenchRunner",
    "longbench_score": "modules.evaluation.longbench_scoring.LongBenchScorer",
    "visualize"      : "modules.evaluation.visualize.VisualizeRunner",
}
```

Runner classes are imported lazily (via `importlib`) so that `--help` works
without loading PyTorch or HuggingFace Transformers.

**CLI override example:**
```bash
python main.py --config configs/eval_parity_base.yaml \
    --override data.prefill_len=2048 data.gen_len=128 run.seed=123
```

---

## 3. Configuration System

### Config hierarchy

Every experiment YAML inherits from `configs/base.yaml` via the `_base_` key:

```yaml
# configs/eval_parity_ours_eager.yaml
_base_: base.yaml
run:
  mode: parity_ours
model:
  attn_implementation: eager
cache:
  backend: windowed
  backend_package: eager
  cache_budget: 0.5
```

`load_config()` in `utils/config.py:484` merges `base.yaml` into each child
config, then validates the result into a typed `ExperimentConfig` dataclass tree.

### Key configuration knobs (`configs/base.yaml`)

| Section | Key | Meaning |
|---|---|---|
| `run` | `mode` | Which runner to invoke |
| `run` | `seed` | Global RNG seed |
| `model` | `name` | HuggingFace model ID |
| `model` | `dtype` | `float16` / `bfloat16` |
| `model` | `attn_implementation` | `eager` or `flash_attention_2` |
| `cache` | `backend` | `dynamic` (full cache) or `windowed` (ours) |
| `cache` | `backend_package` | `eager`, `flash_attn`, or `null` |
| `cache` | `cache_budget` | Float in `(0, 1]` — fraction of prefill tokens to keep |
| `cache` | `window_size` | Tokens per eviction window |
| `cache` | `num_sink_tokens` | Tokens always kept (immutable prefix) |
| `cache` | `local_window_size` | Recent tokens always kept (int or ratio) |
| `data` | `dataset` | `wikitext-103` or `pg19` |
| `data` | `prefill_len` | Tokens given as prompt |
| `data` | `gen_len` | Tokens to generate |
| `data` | `num_samples` | Articles to evaluate |
| `telemetry` | `output_dir` | Directory for all outputs |

### Backend pairing enforcement

`utils/cache_factory.py:88` — `validate_backend_attn_pairing()`:

| `backend_package` | Must pair with `attn_implementation` |
|---|---|
| `flash_attn` | `flash_attention_2` |
| `eager` | `eager` |

Violated pairings raise a `ValueError` before any model is loaded.

---

## 4. The Two Cache Backends

Both backends are structurally identical with one difference: how they extract
attention weights.

### `modules/windowed_cache/` — Flash-attention backend

Used when `backend_package: flash_attn` and `attn_implementation: flash_attention_2`.

Flash attention does not expose the attention matrix, so `hooks.py` registers a
`forward_hook` that recomputes the post-RoPE query from the layer inputs, reads
the keys from the cache, and runs a separate (causally-masked) auxiliary SDPA
call to reconstruct attention weights.

### `modules/windowed_eager_cache/` — Eager backend

Used when `backend_package: eager` and `attn_implementation: eager`.

Eager attention materializes `attn_weights` and includes them in the output tuple.
`hooks.py` registers a plain `register_forward_hook` and reads them directly.
The runner must pass `output_attentions=True` to `model.generate()`.

**When to use which:**  
Use eager on Kaggle T4/P100 or any machine without `flash-attn` installed.  
Use flash-attn2 on A100/H100 for maximum throughput.

---

## 5. Evaluation Suites and Their Outputs

### Suite A — Parity (`parity_base` / `parity_ours`)

**Purpose:** Does our windowed cache select the same top-K windows as an
unrestricted full-cache model?

**Producers:**
- `modules/evaluation/base_parity_runner.py` — full-cache reference run
- `modules/evaluation/ours_parity_runner.py` — windowed cache, teacher-forced

**Outputs:**
```
outputs/
  parity_base_<hash>.npz        NPZ with top_window_indices, window_scores, generated_tokens
  parity_base_<hash>.meta.json  Sidecar with metadata
  parity_ours_eager_<hash>.npz  Same schema + retained_window_ids, retained_window_scores
  parity_ours_eager_<hash>.meta.json
```

The ours run is **teacher-forced**: it decodes the same token sequence as the base
run, so any cache quality difference is isolated and not confounded by diverging
generation.

---

### Suite B — Faithfulness (`faithfulness`)

**Purpose:** How well does our windowed cache's attention score distribution
match the full-cache distribution over the retained windows?

**Producer:** `modules/evaluation/faithfulness_runner.py`  
**Input:** Both parity NPZ files (no model loaded).

**Outputs:**
```
outputs/
  faithfulness_results.npz    Per-(step, layer) metrics: jaccard, cos_sim, pearson,
                               spearman, kl_ours_base, mass_ratio, heterogeneity
  faithfulness_results.meta.json
```

---

### Suite C — Performance (`perf`)

**Purpose:** Measure latency and throughput of prefill and generation with
windowed cache vs full cache.

**Producer:** `modules/evaluation/perf_runner.py`  
**Output:** `outputs/perf_prefill_*.npz`

---

### Suite D — LongBench (`longbench` + `longbench_score`)

**Purpose:** End-task quality: does windowed cache preserve task accuracy on the
LongBench benchmark?

**Producers:**
- `modules/evaluation/longbench_runner.py` — generation
- `modules/evaluation/longbench_scoring.py` — post-hoc metric computation

**Outputs:**
```
outputs/longbench/
  full_cache/
    <dataset>.jsonl             One JSON per example: {pred, answers, all_classes}
    scores.csv                  Per-dataset scores
    run.env                     git commit + pip freeze snapshot
  ours_eager_compression_0.8/
    <dataset>.jsonl
    scores.csv
    run.env
  comparison.csv                Cross-run comparison table
```

---

## 6. Bash Scripts — What to Run and When

All scripts share the same environment setup:
```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8    # deterministic CUDA ops
```

Each script also snapshots `git rev-parse HEAD` and `pip freeze` to a `.env` file
alongside its outputs for full reproducibility.

---

### Run everything at once

```bash
bash scripts/reproduce_all.sh
# CUDA_VISIBLE_DEVICES=1 bash scripts/reproduce_all.sh   # pick a specific GPU
```

Runs all suites in dependency order. Expected runtime: several hours on A100.
Flash-attention variants are commented out by default — uncomment lines 33-34
and 56-57 if `flash-attn` is installed.

---

### Suite A — Step 1: Base run (required first)

```bash
bash scripts/run_parity_base.sh
```

- Config: `configs/eval_parity_base.yaml`
- Model: full DynamicCache, no eviction
- Produces: `outputs/parity_base_*.npz`

**Must run before any ours parity run.** The ours runner loads this NPZ to
teacher-force tokens and validate corpus alignment.

---

### Suite A — Step 2: Ours run

**Eager (always works, no extra dependencies):**
```bash
bash scripts/run_parity_ours_eager.sh
```
- Config: `configs/eval_parity_ours_eager.yaml`
- Produces: `outputs/parity_ours_eager_*.npz`

**Flash-attention2 (A100/H100 only, requires `flash-attn` package):**
```bash
bash scripts/run_parity_ours_flash.sh
```
- Config: `configs/eval_parity_ours_flash.yaml`
- Produces: `outputs/parity_ours_flash_*.npz`

---

### Suite B — Faithfulness (no model, runs in seconds)

```bash
bash scripts/run_faithfulness.sh
```
- Config: `configs/eval_faithfulness.yaml`
- Requires: both parity NPZ files to exist
- Produces: `outputs/faithfulness_results.npz`

Can be re-run without re-doing the parity runs.

---

### Suite C — Performance benchmarks

```bash
bash scripts/run_perf.sh
```
- Config: `configs/eval_perf.yaml`
- Produces: `outputs/perf_prefill_*.npz`

---

### Visualization

```bash
bash scripts/run_visualize.sh
```
- Config: `configs/eval_visualize.yaml`
- Requires: faithfulness NPZ
- Produces: figures in `outputs/`

---

### Suite D — LongBench (Step 1: generate)

**Full-cache baseline:**
```bash
bash scripts/run_longbench_full_cache.sh
```
- Config: `configs/longbench_full_cache.yaml`
- Produces: `outputs/longbench/full_cache/`

**Ours — eager:**
```bash
bash scripts/run_longbench_ours_eager.sh
```
- Config: `configs/longbench_ours_eager.yaml`
- Produces: `outputs/longbench/ours_eager_compression_0.8/`

**Ours — flash-attn2:**
```bash
bash scripts/run_longbench_ours_flash_attn.sh
```
- Config: `configs/longbench_ours_flash_attn.yaml`

---

### Suite D — LongBench (Step 2: score)

```bash
bash scripts/score_longbench.sh
```

No model required. Loops over every directory under `outputs/longbench/`,
calls `longbench_scoring.py` on each, then builds a cross-run comparison table
at `outputs/longbench/comparison.csv`.

Two-stage flow:
1. Score each run directory individually → `<run_dir>/scores.csv`
2. Build comparison: `--baseline full_cache/scores.csv --variants ours_*/scores.csv`

---

## 7. Dependency Order

```
run_parity_base.sh
        │
        ▼
run_parity_ours_eager.sh  (or flash variant)
        │
        ▼
run_faithfulness.sh          (can re-run alone; reads NPZs only)
run_visualize.sh             (reads faithfulness NPZ)
        │
run_longbench_full_cache.sh  (independent — no parity dependency)
run_longbench_ours_eager.sh  (independent — no parity dependency)
        │
        ▼
score_longbench.sh           (post-hoc, reads jsonl outputs only)
```

`run_perf.sh` is fully independent and can run at any point.

---

## 8. Key Utility Modules

### `utils/config.py` — Configuration loading

- `load_config(path, overrides)` (line 484): YAML → nested dict → typed dataclass.
  Handles `_base_` inheritance and dot-notation CLI overrides.
- `ExperimentConfig`: top-level container for all typed sub-configs.
- `validate_parity_pair()` (line 545): cross-checks base/ours metadata before
  faithfulness runner starts.

### `utils/cache_factory.py` — Backend selection

- `get_cache_classes(backend_package)` (line 28): returns
  `(WindowedCache, WindowedCacheConfig, install_score_hooks)` for the chosen backend.
  Uses lazy imports so eager never loads flash_attn and vice versa.
- `validate_backend_attn_pairing()` (line 88): enforces the flash↔flash / eager↔eager rule.

### `utils/metrics.py` — Pure metric functions

- `jaccard_topk()` — vectorized Jaccard similarity of top-K window sets
- `aggregate_per_layer()`, `aggregate_global()` — aggregation helpers
- `final_step_heterogeneity()` — std across heads at last step

### `utils/hashing.py` — Reproducibility fingerprints

- `sha256_file(path)` — SHA-256 of a file on disk
- `sha256_text(text)` — SHA-256 of an article string

These are embedded in every NPZ's `metadata_json` so any downstream runner
can verify it is reading the exact same data it expects.

### `utils/seed.py`

- `seed_everything(seed)`: seeds Python `random`, `numpy`, and `torch`
  (including CUDA) for fully deterministic runs.

---

## 9. Output File Conventions

| File | Contents |
|---|---|
| `parity_base_*.npz` | Reference: `top_window_indices`, `window_scores`, `generated_tokens`, `eviction_step_mask`, `metadata_json` |
| `parity_ours_*.npz` | Same + `retained_window_ids`, `retained_window_scores` |
| `faithfulness_results.npz` | `jaccard`, `jaccard_per_layer`, `jaccard_global`, `heterogeneity`, `cos_sim`, `pearson`, `spearman`, `kl_ours_base`, `mass_ratio`, `metadata_json` |
| `*.meta.json` | Sidecar with SHA checksums and run provenance |
| `*.env` | `git rev-parse HEAD` + `pip freeze` for reproducibility |
| `longbench/<run>/<dataset>.jsonl` | One JSON per example: `{pred, answers, all_classes}` |
| `longbench/<run>/scores.csv` | Per-dataset score (0–100) |
| `longbench/comparison.csv` | Cross-run comparison table |

---

## 10. Adding a New Experiment

1. Create a YAML in `configs/` that sets `_base_: base.yaml` and overrides the
   fields you want to change.
2. If you need a new mode, add a runner class under `modules/evaluation/` and
   register it in `_RUNNER_REGISTRY` in `main.py:25`.
3. Write a one-liner bash script in `scripts/` that calls
   `python main.py --config configs/<your_config>.yaml "$@"`.
4. For custom overrides at call time:
   ```bash
   bash scripts/run_parity_base.sh --override data.num_samples=10 run.seed=99
   ```
   The `"$@"` pass-through in every script forwards extra arguments to `main.py`.
