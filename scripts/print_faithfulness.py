"""Pretty-print faithfulness results (schema v2.0).

Usage (Kaggle cell):
    exec(open("scripts/print_faithfulness.py").read())
  or:
    %run scripts/print_faithfulness.py
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────
NPZ_PATH = "/kaggle/working/outputs/faithfulness/faithfulness_results.npz"

# ── Load ──────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH, allow_pickle=True)

jaccard_global    = data["jaccard_global"]       # [T]
jaccard_per_layer = data["jaccard_per_layer"]    # [T, L]
het               = data["heterogeneity"]        # [L]

# New distribution-comparison metrics — all [T, L]
cos_sim     = data["cos_sim"]
pearson     = data["pearson"]
spearman    = data["spearman"]
kl          = data["kl_ours_base"]
mass_ratio  = data["mass_ratio"]

num_steps  = jaccard_global.shape[0]
num_layers = jaccard_per_layer.shape[1]

meta = {}
if "metadata_json" in data.files:
    meta = json.loads(str(data["metadata_json"][0]))
prefill_len = meta.get("prefill_len", None)
gen_len     = meta.get("gen_len", None)

PREFILL_STEPS = 1
GEN_START     = PREFILL_STEPS
GEN_END       = num_steps

W = 76  # print width

# ══════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════
print()
print("═" * W)
print("  FAITHFULNESS  RESULTS  SUMMARY".center(W))
print("═" * W)

if meta:
    model  = meta.get("model_name", "—")
    ws     = meta.get("window_size", "—")
    sink   = meta.get("num_sink_tokens", "—")
    local  = meta.get("local_window_size_resolved", "—")
    topk   = meta.get("top_k_windows", "—")
    budget = meta.get("cache_budget", "—")
    schema = meta.get("schema_version", "?")
    print(f"  Model          : {model}")
    print(f"  Prefill tokens : {prefill_len}    Gen tokens: {gen_len}")
    print(f"  Window size    : {ws}    Sink: {sink}    Local: {local}")
    print(f"  Top-K windows  : {topk}    Budget: {budget}")
    print(f"  Total steps    : {num_steps}  (1 prefill + {num_steps - 1} gen)"
          f"    Layers: {num_layers}    Schema: v{schema}")
print("─" * W)


def _stats(arr: np.ndarray, label: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{'Phase':<14} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print(f"{pad}{'─'*14} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    pre = arr[:PREFILL_STEPS]
    gen = arr[GEN_START:GEN_END]
    all_ = arr
    if len(pre):
        print(f"{pad}{'Prefill':<14} {pre.mean():>8.4f}  {pre.min():>8.4f}  {pre.max():>8.4f}  {pre.std():>8.4f}")
    if len(gen):
        print(f"{pad}{'Generation':<14} {gen.mean():>8.4f}  {gen.min():>8.4f}  {gen.max():>8.4f}  {gen.std():>8.4f}")
    print(f"{pad}{'Overall':<14} {all_.mean():>8.4f}  {all_.min():>8.4f}  {all_.max():>8.4f}  {all_.std():>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# 1. Jaccard
# ══════════════════════════════════════════════════════════════════════
print()
print("┌" + "─" * (W - 2) + "┐")
print("│" + "  JACCARD  SIMILARITY  (Top-K Window Overlap)".center(W - 2) + "│")
print("└" + "─" * (W - 2) + "┘")
print()
_stats(jaccard_global, "Jaccard")

# per-layer Jaccard
jac_pre = jaccard_per_layer[:PREFILL_STEPS]
jac_gen = jaccard_per_layer[GEN_START:GEN_END]
print()
print(f"  {'':>8}  ┌{'─ Prefill ─':─^18}┐  ┌{'─ Generation ─':─^18}┐  ┌{'─ Overall ─':─^18}┐")
print(f"  {'Layer':>8}  {'Mean':>8}  {'Min':>6}      {'Mean':>8}  {'Min':>6}      {'Mean':>8}  {'Std':>6}")
print(f"  {'─'*8}  {'─'*8}  {'─'*6}      {'─'*8}  {'─'*6}      {'─'*8}  {'─'*6}")
for li in range(num_layers):
    pre_v = jac_pre[:, li] if len(jac_pre) else np.array([0.0])
    gen_v = jac_gen[:, li] if len(jac_gen) else np.array([0.0])
    all_v = jaccard_per_layer[:, li]
    bar = "█" * int(round(all_v.mean() * 8)) + "░" * (8 - int(round(all_v.mean() * 8)))
    print(f"  {li:>8d}  {pre_v.mean():>8.4f}  {pre_v.min():>6.4f}  "
          f"    {gen_v.mean():>8.4f}  {gen_v.min():>6.4f}  "
          f"    {all_v.mean():>8.4f}  {all_v.std():>6.4f}  {bar}")
print(f"  {'─'*8}  {'─'*8}  {'─'*6}      {'─'*8}  {'─'*6}      {'─'*8}  {'─'*6}")
print(f"  {'avg':>8s}  {jac_pre.mean():>8.4f}  {jac_pre.min():>6.4f}  "
      f"    {jac_gen.mean():>8.4f}  {jac_gen.min():>6.4f}  "
      f"    {jaccard_per_layer.mean():>8.4f}  {jaccard_per_layer.std():>6.4f}")


# ══════════════════════════════════════════════════════════════════════
# 2. Distribution Comparison Metrics (new, per [step, layer])
# ══════════════════════════════════════════════════════════════════════
print()
print("┌" + "─" * (W - 2) + "┐")
print("│" + "  DISTRIBUTION  COMPARISON  (Ours vs Base on Retained Windows)".center(W - 2) + "│")
print("└" + "─" * (W - 2) + "┘")
print()
print("  Metrics are mean over layers, then split by Prefill vs Generation.\n")

metrics = [
    ("cos_sim",    cos_sim,    "Cosine Similarity     (↑ higher = better, max 1.0)"),
    ("pearson",    pearson,    "Pearson Correlation   (↑ higher = better, max 1.0)"),
    ("spearman",   spearman,   "Spearman Correlation  (↑ higher = better, max 1.0)"),
    ("kl_ours_base", kl,       "KL Divergence (ours‖base)  (↓ lower = better, min 0)"),
    ("mass_ratio", mass_ratio, "Mass Ratio (base/ours)     (≈1 = well-matched)"),
]

print(f"  {'Metric':<34}  {'Overall':>8}  {'Prefill':>8}  {'Gen':>8}  {'Std(gen)':>8}")
print(f"  {'─'*34}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
for key, arr, label in metrics:
    # mean over layers → [T]
    arr_l = arr.mean(axis=1)
    pre   = arr_l[:PREFILL_STEPS]
    gen   = arr_l[GEN_START:GEN_END]
    print(f"  {label:<34}  {arr_l.mean():>8.4f}  "
          f"{pre.mean() if len(pre) else 0:>8.4f}  "
          f"{gen.mean() if len(gen) else 0:>8.4f}  "
          f"{gen.std()  if len(gen) else 0:>8.4f}")

# Per-layer breakdown for each metric
for key, arr, label in metrics:
    print()
    print(f"  ── {label}")
    print(f"  {'Layer':>6}  {'Overall':>8}  {'Prefill':>8}  {'Gen mean':>8}  {'Gen std':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    pre_l = arr[:PREFILL_STEPS]
    gen_l = arr[GEN_START:GEN_END]
    for li in range(num_layers):
        a_li  = arr[:, li]
        pre_v = pre_l[:, li].mean() if len(pre_l) else 0.0
        gen_v = gen_l[:, li].mean() if len(gen_l) else 0.0
        gen_s = gen_l[:, li].std()  if len(gen_l) else 0.0
        print(f"  {li:>6d}  {a_li.mean():>8.4f}  {pre_v:>8.4f}  {gen_v:>8.4f}  {gen_s:>8.4f}")
    print(f"  {'avg':>6s}  {arr.mean():>8.4f}  "
          f"{pre_l.mean() if len(pre_l) else 0:>8.4f}  "
          f"{gen_l.mean() if len(gen_l) else 0:>8.4f}  "
          f"{gen_l.std()  if len(gen_l) else 0:>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# 3. Head Heterogeneity
# ══════════════════════════════════════════════════════════════════════
print()
print("┌" + "─" * (W - 2) + "┐")
print("│" + "  HEAD  HETEROGENEITY  (Std across heads at final step)".center(W - 2) + "│")
print("└" + "─" * (W - 2) + "┘")
print()
het_max_layer = het.argmax()
print(f"  Mean across layers : {het.mean():.4f}")
print(f"  Most heterogeneous : layer {het_max_layer} (std = {het[het_max_layer]:.4f})")
print()
cols = 4
rows = (num_layers + cols - 1) // cols
for r in range(rows):
    parts = []
    for c in range(cols):
        li = r + c * rows
        if li < num_layers:
            marker = " ◄" if li == het_max_layer else "  "
            parts.append(f"  L{li:<3d} {het[li]:.4f}{marker}")
    print("".join(parts))


# ══════════════════════════════════════════════════════════════════════
# 4. Per-Sample Breakdown
# ══════════════════════════════════════════════════════════════════════
if "per_sample_jaccard_global" in data.files:
    psjg = data["per_sample_jaccard_global"]  # [S, T]
    n_s  = psjg.shape[0]
    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + f"  PER-SAMPLE  BREAKDOWN  ({n_s} samples)".center(W - 2) + "│")
    print("└" + "─" * (W - 2) + "┘")
    print()
    print(f"  {'Sample':>8}  {'Jac overall':>12}  {'Jac prefill':>12}  {'Jac gen':>10}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}")
    for s in range(n_s):
        sj = psjg[s]
        print(f"  {s:>8d}  {sj.mean():>12.4f}  "
              f"{sj[:PREFILL_STEPS].mean():>12.4f}  "
              f"{sj[GEN_START:].mean() if sj.shape[0]>PREFILL_STEPS else 0:>10.4f}")


# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
print()
print("═" * W)
print(f"  Source: {NPZ_PATH}")
print(f"  Arrays: {', '.join(data.files)}")
print("═" * W)
print()
