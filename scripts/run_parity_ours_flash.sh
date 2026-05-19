#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Run Parity Suite — Ours (flash-attn backend, Ampere+ required)
# ============================================================================
# Depends on: outputs/parity_base_*.npz (run run_parity_base.sh first)
# Produces:   outputs/parity_ours_flash_*.npz + .meta.json sidecar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p outputs
git rev-parse HEAD > outputs/parity_ours_flash.env 2>/dev/null || echo "no_git" > outputs/parity_ours_flash.env
pip freeze >> outputs/parity_ours_flash.env 2>/dev/null || true

python "$PROJECT_ROOT/main.py" \
    --config "$PROJECT_ROOT/configs/eval_parity_ours_flash.yaml" \
    "$@"
