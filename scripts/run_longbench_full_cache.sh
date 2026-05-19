#!/usr/bin/env bash
# scripts/run_longbench_full_cache.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run LongBench — Full Cache Baseline
# ============================================================================
# Runs all 16 English datasets with full DynamicCache (no eviction).
# Produces: outputs/longbench/full_cache/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs/longbench/full_cache
git rev-parse HEAD > outputs/longbench/full_cache/run.env 2>/dev/null || echo "no_git" > outputs/longbench/full_cache/run.env
pip freeze >> outputs/longbench/full_cache/run.env

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/longbench_full_cache.yaml" \
    "$@"
