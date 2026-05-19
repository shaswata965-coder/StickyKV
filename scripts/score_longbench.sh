#!/usr/bin/env bash
# scripts/score_longbench.sh
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # kept for consistency across all scripts

# ============================================================================
# Score LongBench — Compute metrics from generated jsonl outputs
# ============================================================================
# Post-hoc scoring: reads jsonls, applies per-dataset metrics, builds
# comparison table.  No model load required.  Re-runnable in seconds.
#
# Two-stage flow (matches DefensiveKV's evaluate.sh):
# 1. Score each run directory individually → scores.csv
# 2. Build cross-run comparison table → comparison.csv + comparison.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Stage 1: Score individual runs
for run_dir in outputs/longbench/*/; do
    if [ -d "$run_dir" ]; then
        echo "Scoring: $run_dir"
        python -m modules.evaluation.longbench_scoring \
            --predictions_dir "$run_dir" \
            --out_csv "${run_dir}scores.csv"
    fi
done

# Stage 2: Build comparison table
echo ""
echo "Building comparison table..."
python -m modules.evaluation.longbench_scoring \
    --out outputs/longbench/comparison.csv \
    --baseline outputs/longbench/full_cache/scores.csv \
    --variants outputs/longbench/ours_*/scores.csv 2>/dev/null || true

echo ""
echo "Done. Results in outputs/longbench/"
