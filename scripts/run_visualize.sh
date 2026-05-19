#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # kept for consistency across all scripts

# ============================================================================
# Run Visualization
# ============================================================================
# Depends on: outputs/*.npz from parity, faithfulness, perf suites
# Produces:   outputs/figures/*.png (+ .pdf if configured)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs/figures

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_visualize.yaml" \
    "$@"
