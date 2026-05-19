#!/usr/bin/env bash
# scripts/run_longbench_ours_flash_attn.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run LongBench — Ours (flash-attn backend)
# ============================================================================
# Windowed KV-cache with flash-attention-2 score extraction.
# Produces: outputs/longbench/ours_flash_attn_compression_0.8/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs/longbench/ours_flash_attn_compression_0.8
git rev-parse HEAD > outputs/longbench/ours_flash_attn_compression_0.8/run.env 2>/dev/null || echo "no_git" > outputs/longbench/ours_flash_attn_compression_0.8/run.env
pip freeze >> outputs/longbench/ours_flash_attn_compression_0.8/run.env

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/longbench_ours_flash_attn.yaml" \
    "$@"
