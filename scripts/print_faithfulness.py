"""Pretty-print faithfulness results with per-layer Jaccard split by Prefill vs Generation.

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

jaccard_global    = data["jaccard_global"]       # [num_steps]
jaccard_per_layer = data["jaccard_per_layer"]    # [num_steps, num_layers]
jaccard_raw       = data["jaccard"]              # [num_steps, num_layers, 1]
lir_proxy         = data["lir_proxy"]            # [num_steps]
het               = data["heterogeneity"]        # [num_layers]

num_steps  = jaccard_global.shape[0]
num_layers = jaccard_per_layer.shape[1]

# Recover prefill_len from metadata to split prefill vs generation steps.
# Step 0 = prefill (all prompt tokens); steps 1..gen_len = generation.
meta = {}
if "metadata_json" in data.files:
    meta = json.loads(str(data["metadata_json"][0]))
prefill_len = meta.get("prefill_len", None)
gen_len     = meta.get("gen_len", None)

# Heuristic: step 0 is the prefill forward pass, steps 1+ are generation.
# This matches the parity runner loop where step==0 feeds the full prompt.
PREFILL_STEPS = 1                          # first step is always prefill
GEN_START     = PREFILL_STEPS
GEN_END       = num_steps

jac_prefill = jaccard_per_layer[:PREFILL_STEPS]   # [1, L]
jac_gen     = jaccard_per_layer[GEN_START:GEN_END] # [gen_steps, L]

lir_prefill = lir_proxy[:PREFILL_STEPS]
lir_gen     = lir_proxy[GEN_START:GEN_END]

W = 72  # print width

# ══════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════
print()
print("═" * W)
print("  FAITHFULNESS  RESULTS  SUMMARY".center(W))
print("═" * W)

if meta:
    model = meta.get("model_name", "—")
    ws    = meta.get("window_size", "—")
    sink  = meta.get("num_sink_tokens", "—")
    local = meta.get("local_window_size_resolved", "—")
    topk  = meta.get("top_k_windows", "—")
    budget = meta.get("cache_budget", "—")
    print(f"  Model          : {model}")
    print(f"  Prefill tokens : {prefill_len}")
    print(f"  Gen tokens     : {gen_len}")
    print(f"  Window size    : {ws}    Sink: {sink}    Local: {local}")
    print(f"  Top-K windows  : {topk}    Budget: {budget}")
    print(f"  Total steps    : {num_steps}  (1 prefill + {num_steps - 1} gen)")
    print(f"  Layers         : {num_layers}")
print("─" * W)

# ══════════════════════════════════════════════════════════════════════
# 1. Jaccard — Global Summary
# ══════════════════════════════════════════════════════════════════════
print()
print("┌" + "─" * (W - 2) + "┐")
print("│" + "  JACCARD  SIMILARITY  (Top-K Window Overlap)".center(W - 2) + "│")
print("└" + "─" * (W - 2) + "┘")

jac_pre_mean = jac_prefill.mean()
jac_gen_mean = jac_gen.mean() if len(jac_gen) > 0 else 0.0
jac_all_mean = jaccard_global.mean()

print()
print(f"  {'Phase':<14} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
print(f"  {'─' * 14} {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
print(f"  {'Prefill':<14} {jac_pre_mean:>8.4f}  {jac_prefill.min():>8.4f}  {jac_prefill.max():>8.4f}  {jac_prefill.std():>8.4f}")
if len(jac_gen) > 0:
    print(f"  {'Generation':<14} {jac_gen_mean:>8.4f}  {jac_gen.min():>8.4f}  {jac_gen.max():>8.4f}  {jac_gen.std():>8.4f}")
print(f"  {'Overall':<14} {jac_all_mean:>8.4f}  {jaccard_global.min():>8.4f}  {jaccard_global.max():>8.4f}  {jaccard_global.std():>8.4f}")

# ══════════════════════════════════════════════════════════════════════
# 2. Jaccard — Per-Layer Breakdown (Prefill vs Generation)
# ══════════════════════════════════════════════════════════════════════
print()
print(f"  {'':>8}  ┌{'─ Prefill ─':─^22}┐  ┌{'─ Generation ─':─^22}┐  ┌{'─ Overall ─':─^22}┐")
print(f"  {'Layer':>8}  {'Mean':>8}  {'Min':>8}      {'Mean':>8}  {'Min':>8}      {'Mean':>8}  {'Std':>8}")
print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}      {'─' * 8}  {'─' * 8}      {'─' * 8}  {'─' * 8}")

for li in range(num_layers):
    pre_vals = jac_prefill[:, li]
    gen_vals = jac_gen[:, li] if len(jac_gen) > 0 else np.array([0.0])
    all_vals = jaccard_per_layer[:, li]

    pre_m = pre_vals.mean()
    pre_mn = pre_vals.min()
    gen_m = gen_vals.mean()
    gen_mn = gen_vals.min()
    all_m = all_vals.mean()
    all_s = all_vals.std()

    # Visual bar (8 chars wide, based on overall mean)
    bar_len = int(round(all_m * 8))
    bar = "█" * bar_len + "░" * (8 - bar_len)

    print(f"  {li:>8d}  {pre_m:>8.4f}  {pre_mn:>8.4f}      {gen_m:>8.4f}  {gen_mn:>8.4f}      {all_m:>8.4f}  {all_s:>8.4f}  {bar}")

# Layer-level summary
pre_layer_means = jac_prefill.mean(axis=0)  # [L]
gen_layer_means = jac_gen.mean(axis=0) if len(jac_gen) > 0 else np.zeros(num_layers)
print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}      {'─' * 8}  {'─' * 8}      {'─' * 8}  {'─' * 8}")
print(f"  {'avg':>8s}  {pre_layer_means.mean():>8.4f}  {pre_layer_means.min():>8.4f}  "
      f"    {gen_layer_means.mean():>8.4f}  {gen_layer_means.min():>8.4f}  "
      f"    {jaccard_per_layer.mean():>8.4f}  {jaccard_per_layer.std():>8.4f}")

# ══════════════════════════════════════════════════════════════════════
# 3. LIR Proxy — Attention Mass Retention
# ══════════════════════════════════════════════════════════════════════
print()
print("┌" + "─" * (W - 2) + "┐")
print("│" + "  LIR  PROXY  (Attention Mass Retention)".center(W - 2) + "│")
print("└" + "─" * (W - 2) + "┘")

print()
print(f"  {'Phase':<14} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
print(f"  {'─' * 14} {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
if len(lir_prefill) > 0:
    print(f"  {'Prefill':<14} {lir_prefill.mean():>8.4f}  {lir_prefill.min():>8.4f}  {lir_prefill.max():>8.4f}  {lir_prefill.std():>8.4f}")
if len(lir_gen) > 0:
    print(f"  {'Generation':<14} {lir_gen.mean():>8.4f}  {lir_gen.min():>8.4f}  {lir_gen.max():>8.4f}  {lir_gen.std():>8.4f}")
print(f"  {'Overall':<14} {lir_proxy.mean():>8.4f}  {lir_proxy.min():>8.4f}  {lir_proxy.max():>8.4f}  {lir_proxy.std():>8.4f}")

# Retention quality indicator
lir_mean = lir_proxy.mean()
if lir_mean >= 0.90:
    quality = "★★★  Excellent — ≥90% attention mass retained"
elif lir_mean >= 0.75:
    quality = "★★☆  Good — ≥75% attention mass retained"
elif lir_mean >= 0.50:
    quality = "★☆☆  Fair — ≥50% attention mass retained"
else:
    quality = "☆☆☆  Poor — <50% attention mass retained"
print(f"\n  Quality: {quality}")

# ══════════════════════════════════════════════════════════════════════
# 4. Head Heterogeneity (final step)
# ══════════════════════════════════════════════════════════════════════
print()
print("┌" + "─" * (W - 2) + "┐")
print("│" + "  HEAD  HETEROGENEITY  (Std across heads at final step)".center(W - 2) + "│")
print("└" + "─" * (W - 2) + "┘")

print()
het_mean = het.mean()
het_max_layer = het.argmax()
print(f"  Mean across layers : {het_mean:.4f}")
print(f"  Most heterogeneous : layer {het_max_layer} (std = {het[het_max_layer]:.4f})")
print()

# Compact per-layer display (4 columns)
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
# 5. Per-Sample Breakdown (if available)
# ══════════════════════════════════════════════════════════════════════
if "per_sample_jaccard_global" in data.files:
    per_sample_jac = data["per_sample_jaccard_global"]  # [num_samples, num_steps]
    per_sample_lir = data["per_sample_lir_proxy"]       # [num_samples, num_steps]
    n_samples = per_sample_jac.shape[0]

    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + f"  PER-SAMPLE  BREAKDOWN  ({n_samples} samples)".center(W - 2) + "│")
    print("└" + "─" * (W - 2) + "┘")

    print()
    print(f"  {'Sample':>8}  {'Jaccard':>10}  {'LIR':>10}  {'Jac (prefill)':>14}  {'Jac (gen)':>12}")
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 14}  {'─' * 12}")
    for s in range(n_samples):
        sj = per_sample_jac[s]
        sl = per_sample_lir[s]
        sj_pre = sj[:PREFILL_STEPS].mean()
        sj_gen = sj[GEN_START:].mean() if sj.shape[0] > PREFILL_STEPS else 0.0
        print(f"  {s:>8d}  {sj.mean():>10.4f}  {sl.mean():>10.4f}  {sj_pre:>14.4f}  {sj_gen:>12.4f}")

# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════
print()
print("═" * W)
print(f"  Source: {NPZ_PATH}")
print(f"  Arrays: {', '.join(data.files)}")
print("═" * W)
print()
