#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run Faithfulness Evaluation (Suite B)
# ============================================================================
# Depends on: outputs/parity_base_*.npz + outputs/parity_ours_*.npz
# Produces:   outputs/faithfulness_results.npz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_faithfulness.yaml" \
    "$@"
