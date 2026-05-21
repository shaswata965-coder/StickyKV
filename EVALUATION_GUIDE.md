# Evaluation Guide — Corpus Loading to Final Scores

This document traces the complete evaluation pipeline: how articles are loaded,
how each evaluation suite runs, and how every metric (Jaccard, cosine similarity,
Pearson, Spearman, KL divergence, mass ratio, LongBench) is computed step by step.

---

## 1. Corpus Loading

### Entry point

Every evaluation suite that needs text begins by constructing a `CorpusLoader`.

**File:** `data/corpus_loader.py`  
**Class:** `CorpusLoader`

```python
loader = CorpusLoader(dataset="wikitext-103")  # or "pg19"
```

### `CorpusLoader.load()` (corpus_loader.py:84)

The first call triggers the actual dataset fetch; subsequent calls return the
cached list.

**wikitext-103 path:**
1. `load_dataset("wikitext", "wikitext-103-raw-v1", split="test")` from HuggingFace.
2. All rows concatenated into one large string.
3. `_split_into_articles()` (line 70): splits on `\n(?= = [^=])` — every level-1
   heading `= Title =` starts a new article. Empty articles are dropped.
4. Returns `List[str]` — one entry per Wikipedia article.

**PG19 path:**
1. `load_dataset("deepmind/pg19", split="test")`.
2. Returns `[row["text"] for row in ds]` — one entry per book.

### `CorpusLoader.sample_articles(n, seed)` (corpus_loader.py:118)

```python
rng = random.Random(seed)
indices = rng.sample(range(len(articles)), n)
indices.sort()          # deterministic ordering regardless of sample order
return [articles[i] for i in indices]
```

The sort ensures that running with the same `(n, seed)` always returns articles
in the same order, making multi-run comparisons reproducible.

### Article identity — `ArticleRegistry`

**File:** `data/article_registry.py`  
**Class:** `ArticleRegistry`

After articles are sampled, each one is registered:

```python
registry.register_article(dataset, article_id, text)
# Computes sha256(text.encode("utf-8"))
# Stores (dataset, article_id) → sha256 mapping
```

The SHA is embedded in every output NPZ's metadata. When `OursParityRunner` loads
a base NPZ it calls `sha256_file()` and cross-checks per-article SHAs so that a
mismatch in corpus version is caught before any GPU work begins.

---

## 2. Suite A — Parity (Baseline Run)

### What it measures

The base parity run is a **reference run** with no eviction. Its purpose is to
record the "ground truth" top-K window selections so that the ours run can be
compared against them.

### Runner

**File:** `modules/evaluation/base_parity_runner.py`  
**Class:** `BaseParityRunner`  
**Method:** `run()` (line 26)

### Step-by-step flow

```
1. Load config, corpus loader, article registry
2. Load model with attn_implementation="eager", DynamicCache (no eviction)
3. For each sample article:
   a. Tokenize (truncate to prefill_len tokens)
   b. Prefill: model(input_ids, past_key_values, output_attentions=True)
   c. For each of gen_len generation steps:
      i.  Forward: model(last_token, past_key_values, output_attentions=True)
      ii. For each layer, collect attn_weights [B, H_q, 1, S]
      iii. Accumulate cumulative attention scores across query steps (H2O style)
      iv. Compute window scores:
            compute_window_scores(attn, num_sink, window_size)
      v.  Select top-K windows from evictable region (topk on evictable slice)
      vi. Store: top_window_indices[step, layer, :] = topk indices
4. Save NPZ:
   - top_window_indices  [num_samples, num_steps, num_layers, K]
   - window_scores       [num_samples, num_steps, num_layers, H_q, W]
   - generated_tokens    [num_samples, num_steps]
   - eviction_step_mask  [num_samples, num_steps]  (all False for base)
   - metadata_json       JSON blob with run parameters + SHA fingerprints
```

### Suite A — Ours Run

**File:** `modules/evaluation/ours_parity_runner.py`  
**Class:** `OursParityRunner`

1. Validates backend/attn_implementation pairing via
   `validate_backend_attn_pairing()` (`utils/cache_factory.py:88`).
2. Loads and validates the base NPZ — checks `article_sha`, `model_name`,
   `window_size`, `num_sink_tokens`, `seed`, `prefill_len`, `gen_len` all match.
3. Loads corpus, cross-checks per-sample SHA.
4. Loads model with chosen `attn_implementation`.
5. Calls `install_score_hooks(model, cache, resolved_config)`.
6. For each sample, runs generation **teacher-forced** from `base["generated_tokens"]`
   (so both runs decode identical token sequences — the only difference is the cache).
7. Saves same NPZ schema as base, plus `retained_window_ids` and
   `retained_window_scores` arrays needed by the faithfulness runner.

---

## 3. Suite B — Faithfulness (Score Distribution Comparison)

**File:** `modules/evaluation/faithfulness_runner.py`  
**Class:** `FaithfulnessRunner`  
**Method:** `run()` (line 85)

This suite loads **no model**. It is pure tensor arithmetic on the two NPZ files.

### Input loading

```python
base = _load_npz(fc.base_npz_path)    # parity_base_*.npz
ours = _load_npz(fc.ours_npz_path)    # parity_ours_*.npz
```

`_load_npz()` (line 65): reads `metadata_json` as a JSON string and all array
arrays from the NPZ into a dict. Metadata alignment is validated before any
metric computation (matching `article_sha`, `seed`, `prefill_len`, `gen_len`,
`window_size`, `num_sink_tokens`, `model_name`).

### Metric computation — `_compute_metrics()` (line 113)

Arrays loaded:
- `base_ws`   — `[S, T, L, H, W]` window scores from base run
- `base_tk`   — `[S, T, L, K]`    top-K window indices from base run
- `ours_tk`   — `[S, T, L, K]`    top-K window indices from ours run
- `ours_rid`  — `[S, T, L, M]`    retained window IDs (mapped back to original positions)
- `ours_rsc`  — `[S, T, L, H, M]` ours scores at retained windows

For each sample `s`, step `t`, layer `li`:

```
retained IDs = ours_rid[s, t, li]          (M values, -1 padded)
valid        = (retained IDs >= 0) & (retained IDs < W_act)
ret_ids      = retained IDs at valid positions    [n_ret]

o_sc = ours_rsc[s, t, li, :, valid].mean(dim=0)  [n_ret]  (mean over heads)
b_sc = base_ws [s, t, li, :, ret_ids].mean(dim=0) [n_ret]  (base scores at same windows)
```

Both vectors are over the **same set of windows** (the ones ours chose to retain,
looked up in the base's score array at their original positions via `ret_ids`).

#### Metric 1 — Jaccard similarity

**File:** `utils/metrics.py:32`  
**Function:** `jaccard_topk(ours_topk, base_topk) → [T, L, H]`

Measures whether ours and base select the same top-K window indices.

```python
ours_exp = ours_topk.unsqueeze(-1)   # [T, L, H, K, 1]
base_exp = base_topk.unsqueeze(-2)   # [T, L, H, 1, K]
matches  = (ours_exp == base_exp)    # [T, L, H, K, K]

# Intersection: for each ours element, does any base element match?
intersection = matches.any(dim=-1).sum(dim=-1).float()  # [T, L, H]

# Union = |A| + |B| - |A∩B| (both sets have exactly K elements)
union    = 2.0 * K - intersection
jaccard  = intersection / union      # [T, L, H] ∈ [0, 1]
```

**Aggregation:**
- `aggregate_per_layer()` (metrics.py:68): `j.mean(dim=-1)` → `[T, L]`
- `aggregate_global()` (metrics.py:73): `j.mean(dim=(-2,-1))` → `[T]`
- `final_step_heterogeneity()` (metrics.py:78): `j[-1].std(dim=-1)` → `[L]`
  (std across heads at the last step — measures per-layer agreement)

#### Metric 2 — Cosine similarity

**File:** `modules/evaluation/faithfulness_runner.py:33`  
**Function:** `_cosine(o_sc, b_sc)`

```python
F.cosine_similarity(o_sc.unsqueeze(0), b_sc.unsqueeze(0), dim=1).clamp(-1, 1)
```

Range: `[-1, 1]`. Higher is better (1 = identical direction in score space).

#### Metric 3 — Pearson correlation

**File:** `faithfulness_runner.py:38`  
**Function:** `_pearson(o_sc, b_sc, eps=1e-8)`

```python
a_c = a - a.mean()
b_c = b - b.mean()
(a_c * b_c).sum() / (a_c.norm() * b_c.norm()).clamp(min=eps)
```

Range: `[-1, 1]`. Measures linear correlation of score magnitudes.

#### Metric 4 — Spearman rank correlation

**File:** `faithfulness_runner.py:45`  
**Function:** `_spearman(o_sc, b_sc)`

```python
a_rank = a.argsort().argsort().float()
b_rank = b.argsort().argsort().float()
_pearson(a_rank, b_rank)
```

Pearson applied to ranks. Robust to monotone non-linearities in score magnitude.
Range: `[-1, 1]`.

#### Metric 5 — KL divergence KL(ours ‖ base)

**File:** `faithfulness_runner.py:52`  
**Function:** `_kl(p=o_sc, q=b_sc, eps=1e-8)`

```python
p = o_sc.clamp(min=0)
q = b_sc.clamp(min=0)
p_prob = (p + eps) / (p.sum() + eps * n)   # normalize to distribution
q_prob = (q + eps) / (q.sum() + eps * n)
kl = (p_prob * (p_prob.log() - q_prob.log())).sum().clamp(min=0)
```

`p` = ours distribution, `q` = base distribution.  
Range: `[0, ∞)`. Lower is better (0 = distributions identical).

#### Metric 6 — Mass ratio

**File:** `faithfulness_runner.py:217`

```python
mr = b_sc.sum() / o_sc.sum().clamp(min=1e-8)
```

Ratio of total attention mass base assigned to the retained windows vs what ours
assigned. `≈1` means ours and base agree on total importance of the retained set.

### Output

```
np.savez_compressed("outputs/faithfulness_results.npz",
    jaccard           [T, L, 1]      per-(step, layer) Jaccard
    jaccard_per_layer [T, L]         mean over heads
    jaccard_global    [T]            mean over heads + layers
    heterogeneity     [L]            std across heads at last step
    cos_sim           [T, L]         cosine similarity
    pearson           [T, L]         Pearson correlation
    spearman          [T, L]         Spearman rank correlation
    kl_ours_base      [T, L]         KL(ours ‖ base)
    mass_ratio        [T, L]         base_mass / ours_mass
    metadata_json     JSON string    provenance + SHA checksums
)
```

---

## 4. Suite B — Legacy Faithfulness Metrics (utils/metrics.py)

These functions exist for historical/alternative faithfulness measures.
They operate on raw attention tensors (not window scores).

### Layer Information Retention — `lir()` (metrics.py:88)

```python
lir(full_attn, retained_positions)
# full_attn:         [S, L, H, max_cache_len]
# retained_positions:[S, L, max_retained]  (-1 padded)

gathered = gather(full_attn, dim=-1, retained_positions)  # [S, L, H, max_retained]
gathered *= valid_mask    # zero out -1-padded slots
lir_val  = gathered.sum(dim=-1)   # [S, L, H]  ∈ [0, 1]
```

**Interpretation:** fraction of base attention mass that falls on retained tokens.
1.0 = all attention is on retained tokens.

### Missed mass — `missed_mass()` (metrics.py:133)

```python
1.0 - lir(full_attn, retained_positions)
```

### KL (alternative form) — `kl_inverse()` (metrics.py:148)

```python
kl_inverse(full_attn, ours_attn, retained_positions)
# Gather base attention at retained positions, renormalize to a sub-distribution.
# Compute KL(ours_attn ‖ base_restricted) over the retained set.
# Returns [S, L, H]
```

### Global LIR — `global_lir()` (metrics.py:205)

```python
per_head_lir.mean(dim=(-2, -1))   # mean over layers and heads → [S]
```

---

## 5. Suite D — LongBench

### Runner

**File:** `modules/evaluation/longbench_runner.py` (not fully shown in exploration,
but orchestrated identically to the parity runners)

**Config:** `LongBenchConfig` in `utils/config.py`  
**Datasets:** Multi-document QA, single-document QA, summarization, code, etc.

### Prediction generation

1. Loads each LongBench dataset from HuggingFace (e.g. `THUDM/LongBench`).
2. For each example: concatenate context + instruction → tokenize.
3. Runs `model.generate()` with either full `DynamicCache` or `WindowedCache`.
4. Decodes prediction and writes `<dataset>.jsonl` to `predictions/` dir.

### Scoring — `score_predictions()` (longbench_scoring.py:78)

**File:** `modules/evaluation/longbench_scoring.py`  
**Class:** `LongBenchScorer`  
**Entry:** `score_predictions(predictions_dir, output_csv)`

```
1. Load dataset2metric.json   (maps each dataset name → metric function name)
2. For each <dataset>.jsonl in predictions_dir:
   a. Load all examples: {pred, answers: List[str], all_classes?: List[str]}
   b. For first-line-only datasets (trec, triviaqa, samsum, lsht):
         pred = pred.split("\n")[0].strip()
   c. score = mean over examples of:
         max(metric_fn(pred, gt, all_classes) for gt in answers)
   d. Write to CSV: dataset, num_examples, score (× 100)
3. Compute category averages across datasets
```

### Metric functions (longbench_metrics.py — vendored from THUDM/LongBench verbatim)

Do not modify these. They are kept identical to the published LongBench codebase
so results are directly comparable to the literature.

| Metric function | Datasets | Method |
|---|---|---|
| `qa_f1_score` (line 150) | hotpotqa, triviaqa, multifieldqa_en, 2wikimqa, musique | Token-level F1 after normalization |
| `qa_f1_zh_score` | multifieldqa_zh | Same, Chinese tokenization |
| `rouge_score` (line ~120) | gov_report, qasper, multi_news, vcsum, trec, samsum, lsht | ROUGE-L F1 |
| `rouge_zh_score` | vcsum, lsht | ROUGE-L on Chinese text |
| `classification_score` | trec, lsht | Match prediction in `all_classes` list |
| `retrieval_score` | passage_count, passage_retrieval_en | Paragraph ID from regex |
| `retrieval_zh_score` | passage_retrieval_zh | Same, Chinese |
| `count_score` | passage_count | Extract digit, compare |
| `code_sim_score` | lcc, repobench-p | Fuzzy string match on code |

**`normalize_answer()` (longbench_metrics.py:24):**
```python
lower → remove articles (a/an/the) → remove punctuation → collapse whitespace
```
Applied before all English QA metrics.

**`f1_score()` (longbench_metrics.py:139):**
```python
# Token-level F1 via Counter intersection
common = Counter(pred_tokens) & Counter(gold_tokens)
intersection = sum(common.values())
precision = intersection / len(pred_tokens)
recall    = intersection / len(gold_tokens)
f1 = (2 * precision * recall) / (precision + recall)  if denom > 0 else 0
```

**`qa_f1_score()` (longbench_metrics.py:150):**
```python
prediction = normalize_answer(prediction)
ground_truth = normalize_answer(ground_truth)
f1_score(prediction.split(), ground_truth.split())
```

---

## 6. Metric Quick Reference

| Suite | Metric | Range | Better |
|---|---|---|---|
| A | Jaccard (top-K window overlap) | [0, 1] | Higher |
| B | Cosine similarity | [-1, 1] | Higher |
| B | Pearson correlation | [-1, 1] | Higher |
| B | Spearman rank correlation | [-1, 1] | Higher |
| B | KL(ours ‖ base) | [0, ∞) | Lower |
| B | Mass ratio (base/ours) | (0, ∞) | ≈ 1 |
| B (legacy) | LIR | [0, 1] | Higher |
| B (legacy) | Missed mass | [0, 1] | Lower |
| D | Dataset-specific (F1/ROUGE/exact) | [0, 100] | Higher |

---

## 7. End-to-End Evaluation Pipeline

```
CorpusLoader.load()
  └─► _load_wikitext103() or _load_pg19()
        └─► HuggingFace datasets
              └─► _split_into_articles()   [wikitext only]

CorpusLoader.sample_articles(n, seed)
  └─► random.Random(seed).sample() + sort

ArticleRegistry.register_article()
  └─► sha256(text) → identity record

BaseParityRunner.run()                     [Suite A base]
  └─► model.generate()  DynamicCache
        └─► attn_weights per step per layer
              └─► compute_window_scores()  scorer.py:17
                    └─► topk on evictable slice
                          └─► save NPZ (top_window_indices, window_scores, …)

OursParityRunner.run()                     [Suite A ours]
  └─► install_score_hooks()
        └─► model.generate()  WindowedCache
              └─► hooks → compute_window_scores() → cache.update() → eviction
                    └─► save NPZ (+ retained_window_ids, retained_window_scores)

FaithfulnessRunner.run()                   [Suite B]
  └─► load base NPZ + ours NPZ
        └─► _validate_alignment()
              └─► _compute_metrics()
                    ├─► jaccard_topk()        metrics.py:32
                    ├─► _cosine()             faithfulness_runner.py:33
                    ├─► _pearson()            faithfulness_runner.py:38
                    ├─► _spearman()           faithfulness_runner.py:45
                    ├─► _kl()                 faithfulness_runner.py:52
                    └─► mass ratio            faithfulness_runner.py:217
                          └─► save faithfulness_results.npz

LongBenchRunner.run()                      [Suite D]
  └─► model.generate() for each example
        └─► write predictions/<dataset>.jsonl

LongBenchScorer.run()                      [Suite D scoring]
  └─► score_predictions()                  longbench_scoring.py:78
        └─► dataset2metric.json → metric function dispatch
              └─► normalize_answer() + f1/rouge/exact  longbench_metrics.py
                    └─► write scores CSV
```
