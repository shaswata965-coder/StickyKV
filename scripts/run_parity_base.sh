#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8       # required for deterministic CUDA

# ============================================================================
# Run Parity Suite — Base (full cache, DynamicCache)
# ============================================================================
# Produces: outputs/parity_base_*.npz + .meta.json sidecar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs
git rev-parse HEAD > outputs/parity_base.env 2>/dev/null || echo "no_git" > outputs/parity_base.env
pip freeze >> outputs/parity_base.env 2>/dev/null || true

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_parity_base.yaml" \
    "$@"
