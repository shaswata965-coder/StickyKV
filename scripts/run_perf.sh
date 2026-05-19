#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run Performance Benchmarks (Suite C)
# ============================================================================
# Produces: outputs/perf_prefill*.npz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs
git rev-parse HEAD > outputs/perf.env 2>/dev/null || echo "no_git" > outputs/perf.env
pip freeze >> outputs/perf.env 2>/dev/null || true

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_perf.yaml" \
    "$@"
