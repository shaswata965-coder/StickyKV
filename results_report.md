# StickyKV Faithfulness Results Report

> **Run:** `/kaggle/working/StickyKV/outputs` · 8 configurations · 615 generation steps · Model 1
> **Date analyzed:** 2026-05-31

---

## Configuration Key

| Symbol | Meaning | Values tested |
|--------|---------|---------------|
| `cb`   | Cache budget (fraction of KV slots) | 0.25, 0.5 |
| `lws`  | Local window size | 32, 64 |
| `ws`   | Sticky window size | 1, 8 |

---

## Section A — Faithfulness Summary

| Config | Jaccard ↑ | CosSim ↑ | Pearson ↑ | Spearman ↑ | KL ↓ | MassRatio ≈1 | GlobalLIR ↓ | MissedMass ↓ |
|--------|----------:|----------:|----------:|-----------:|-----:|-------------:|------------:|-------------:|
| cb0.25 · lws32 · **ws1** | 0.5807 | 0.7198 | 0.6046 | 0.9001 | 0.3753 | 0.5658 | 0.0114 | 0.6773 |
| cb0.25 · lws32 · **ws8** | 0.7332 | 0.9820 | 0.9653 | 0.9528 | 0.0229 | 0.7567 | 0.0092 | 0.7227 |
| cb0.25 · lws64 · **ws1** | 0.5639 | 0.7195 | 0.6304 | 0.9429 | 0.3934 | 0.5380 | 0.0052 | 0.7348 |
| cb0.25 · lws64 · **ws8** | 0.6611 | 0.9890 | 0.9793 | 0.9805 | 0.0213 | 0.8163 | 0.0052 | 0.7757 |
| cb0.5 · lws32 · **ws1** | 0.5954 | 0.7116 | 0.6286 | 0.9117 | 0.3371 | 0.6589 | 0.0292 | 0.4843 |
| cb0.5 · lws32 · **ws8** | 0.6957 | 0.9749 | 0.9538 | 0.9371 | 0.0275 | 0.8318 | 0.0351 | 0.5328 |
| cb0.5 · lws64 · **ws1** | 0.5970 | 0.7106 | 0.6312 | 0.9223 | 0.3435 | 0.6555 | 0.0202 | 0.5097 |
| cb0.5 · lws64 · **ws8** | **0.7473** | **0.9900** | **0.9777** | **0.9712** | **0.0126** | **0.8707** | 0.0268 | 0.5588 |

> Bold = best value per column. All configs ran with prefill=409, gen=615, seed=42.

---

## Section B — Trend Analysis

### Trend 1 · Window Size (ws1 → ws8) is the dominant driver

Window size has the largest single effect of any parameter.

| Pair | Jaccard ws1 | Jaccard ws8 | Δ Jaccard | KL ws1 | KL ws8 | Δ KL |
|------|------------:|------------:|----------:|-------:|-------:|-----:|
| cb0.25 · lws32 | 0.5807 | 0.7332 | **+0.153** | 0.3753 | 0.0229 | **−0.353** |
| cb0.25 · lws64 | 0.5639 | 0.6611 | **+0.097** | 0.3934 | 0.0213 | **−0.372** |
| cb0.5 · lws32  | 0.5954 | 0.6957 | **+0.100** | 0.3371 | 0.0275 | **−0.310** |
| cb0.5 · lws64  | 0.5970 | 0.7473 | **+0.150** | 0.3435 | 0.0126 | **−0.321** |

- **Jaccard** gains 0.10–0.15 universally when switching from ws=1 to ws=8.
- **Cosine similarity** jumps from ~0.71 to ~0.98–0.99 — a near-perfect correlation with the base distribution.
- **Pearson and Spearman** both leap from the 0.60–0.63 range to 0.95–0.98.
- **KL divergence** collapses by roughly 15× (from ~0.35 to ~0.02), indicating the selected key distribution almost matches the baseline.
- **Mass ratio** improves from ~0.56–0.66 to ~0.76–0.87.
- **Trade-off:** MissedMass *increases* slightly with ws8 (~+0.04–0.05), meaning slightly more token-level loss per flush, but the distribution-level faithfulness improvement overwhelmingly outweighs this.

---

### Trend 2 · Cache Budget (cb0.25 → cb0.5) most strongly reduces Missed Mass

| Pair | MissedMass cb0.25 | MissedMass cb0.5 | Δ | GlobalLIR cb0.25 | GlobalLIR cb0.5 | Δ |
|------|------------------:|-----------------:|--:|-----------------:|----------------:|--:|
| lws32 · ws1 | 0.6773 | 0.4843 | **−0.193** | 0.0114 | 0.0292 | **+0.018** |
| lws32 · ws8 | 0.7227 | 0.5328 | **−0.190** | 0.0092 | 0.0351 | **+0.026** |
| lws64 · ws1 | 0.7348 | 0.5097 | **−0.225** | 0.0052 | 0.0202 | **+0.015** |
| lws64 · ws8 | 0.7757 | 0.5588 | **−0.217** | 0.0052 | 0.0268 | **+0.022** |

- Doubling the cache budget from 0.25 → 0.5 cuts missed mass by ~0.19–0.23 across all settings — a 25–30% reduction.
- **Trade-off:** GlobalLIR (policy instability) rises consistently when the budget is larger, roughly 3–4× higher LIR at cb=0.5. The policy rescues tokens more often, which is expected with a larger budget competing for more slots.
- Effect on Jaccard/CosSim/Pearson/Spearman is modest (±0.01–0.05), so the faithfulness gains are primarily in the mass-retention dimension, not distribution alignment.

---

### Trend 3 · Local Window Size (lws32 → lws64) shows conditional improvement

| Pair | Jaccard lws32 | Jaccard lws64 | Δ | GlobalLIR lws32 | GlobalLIR lws64 | Δ |
|------|-------------:|-------------:|--:|----------------:|----------------:|--:|
| cb0.25 · ws1 | 0.5807 | 0.5639 | −0.017 | 0.0114 | 0.0052 | **−0.006** |
| cb0.25 · ws8 | 0.7332 | 0.6611 | −0.072 | 0.0092 | 0.0052 | **−0.004** |
| cb0.5 · ws1  | 0.5954 | 0.5970 | +0.002 | 0.0292 | 0.0202 | **−0.009** |
| cb0.5 · ws8  | 0.6957 | 0.7473 | **+0.052** | 0.0351 | 0.0268 | **−0.008** |

- Larger local window (lws=64) consistently **lowers GlobalLIR** — the policy becomes more stable regardless of other settings.
- Effect on Jaccard is inconsistent: it decreases at cb=0.25 but improves at cb=0.5. This suggests local window size interacts with cache budget non-linearly.
- The best Jaccard overall (0.7473) is achieved at **cb0.5 + lws64 + ws8** — all three parameters at their larger value.

---

### Trend 4 · Early vs. Late Phase — Faithfulness decays with generation depth

Across every single configuration, the early generation phase (steps 0–152) outperforms the late phase (steps 153–614) on every metric.

| Config | Jaccard Early | Jaccard Late | Δ | KL Early | KL Late | Δ |
|--------|-------------:|-------------:|--:|----------:|--------:|--:|
| cb0.25 · lws32 · ws1 | 0.7770 | 0.5157 | −0.261 | 0.1239 | 0.4586 | +0.335 |
| cb0.25 · lws32 · ws8 | 0.9024 | 0.6771 | −0.225 | 0.0030 | 0.0295 | +0.027 |
| cb0.5  · lws64 · ws8 | 0.9460 | 0.6815 | −0.265 | 0.0024 | 0.0160 | +0.014 |

- Every config starts with Jaccard ~0.78–0.95 in the early phase and degrades to ~0.49–0.75 by the late phase.
- KL divergence worsens dramatically in the late phase for ws=1 configs (+0.33), and more mildly for ws=8 (+0.01–0.03).
- **Interpretation:** the Sticky-K policy tracks the base attention distribution well at the start of generation but drifts as sequence length grows — a natural effect of the KV cache budget becoming proportionally smaller relative to full context.

---

### Trend 5 · Layer 1 is a structural outlier

Across all 8 configurations, Layer 1 consistently exhibits:

- The **lowest Jaccard** of any layer (range: 0.23–0.74, versus 0.35–1.00 for all other layers)
- The **highest LIR** of any layer (range: 0.012–0.097, versus 0.000–0.050 for other layers)
- The **lowest MissedMass** per flush — Layer 1 retains more token mass but has the least stable key-selection policy

This makes Layer 1 the most structurally divergent layer in the model; its attention pattern changes most unpredictably between steps, causing the Sticky-K policy to rescind keys more frequently.

---

### Trend 6 · Head Heterogeneity is zero throughout

All configurations report `mean het = 0.0000` with zero standard deviation across all 16 layers at the final step. This means all attention heads within each layer select identical key sets under the Sticky-K policy — heads are fully synchronized and there is no within-layer divergence to exploit or worry about.

---

## Section C — Configuration Rankings

### By overall faithfulness (Jaccard)
| Rank | Config | Jaccard |
|------|--------|--------:|
| 1 | cb0.5 · lws64 · ws8 | **0.7473** |
| 2 | cb0.25 · lws32 · ws8 | 0.7332 |
| 3 | cb0.5 · lws32 · ws8 | 0.6957 |
| 4 | cb0.25 · lws64 · ws8 | 0.6611 |
| 5 | cb0.5 · lws64 · ws1 | 0.5970 |
| 6 | cb0.5 · lws32 · ws1 | 0.5954 |
| 7 | cb0.25 · lws32 · ws1 | 0.5807 |
| 8 | cb0.25 · lws64 · ws1 | 0.5639 |

### By policy stability (GlobalLIR — lower is better)
| Rank | Config | GlobalLIR |
|------|--------|----------:|
| 1 | cb0.25 · lws64 · ws1 | **0.0052** |
| 1 | cb0.25 · lws64 · ws8 | **0.0052** |
| 3 | cb0.25 · lws32 · ws8 | 0.0092 |
| 4 | cb0.25 · lws32 · ws1 | 0.0114 |
| 5 | cb0.5 · lws64 · ws1 | 0.0202 |
| 6 | cb0.5 · lws64 · ws8 | 0.0268 |
| 7 | cb0.5 · lws32 · ws1 | 0.0292 |
| 8 | cb0.5 · lws32 · ws8 | **0.0351** |

### By token retention (MissedMass — lower is better)
| Rank | Config | MissedMass |
|------|--------|-----------:|
| 1 | cb0.5 · lws32 · ws1 | **0.4843** |
| 2 | cb0.5 · lws64 · ws1 | 0.5097 |
| 3 | cb0.5 · lws32 · ws8 | 0.5328 |
| 4 | cb0.5 · lws64 · ws8 | 0.5588 |
| 5 | cb0.25 · lws32 · ws1 | 0.6773 |
| 6 | cb0.25 · lws32 · ws8 | 0.7227 |
| 7 | cb0.25 · lws64 · ws1 | 0.7348 |
| 8 | cb0.25 · lws64 · ws8 | **0.7757** |

---

## Section D — Key Takeaways

| # | Finding |
|---|---------|
| 1 | **Window size (ws) is the single most impactful parameter.** ws=8 delivers +0.10–0.15 Jaccard and ~15× lower KL versus ws=1. |
| 2 | **Cache budget (cb) primarily controls token retention, not distribution alignment.** cb=0.5 cuts missed mass by ~25% but raises LIR by 3–4×. |
| 3 | **Larger local window (lws=64) consistently stabilizes the policy** (lower GlobalLIR) but only improves Jaccard at the higher budget (cb=0.5). |
| 4 | **Faithfulness degrades with generation depth in all settings.** The early phase (first ~25% of steps) is consistently 0.15–0.30 Jaccard points better than the late phase. |
| 5 | **Layer 1 is anomalous** — highest LIR, lowest Jaccard — and may benefit from special treatment (e.g., per-layer budget allocation or a larger local window). |
| 6 | **Best all-round config is cb0.5 · lws64 · ws8** (top Jaccard 0.747, top MassRatio 0.871, lowest KL 0.013). If policy stability is the priority, **cb0.25 · lws64 · ws8** is preferred (GlobalLIR 0.0052 vs 0.0268 with only modest faithfulness cost). |
| 7 | **Head heterogeneity = 0 throughout** — no within-layer head divergence; the Sticky-K policy is perfectly head-uniform. |

---

*Generated from `C:\Study Tracker\results.md` — StickyKV faithfulness NPZ analysis.*
