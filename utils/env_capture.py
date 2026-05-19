"""Environment capture for reproducibility metadata.

Returns a dict with library versions, GPU info, and git commit SHA.
flash_attn is **never** imported if it is not installed — the import is
fully guarded so the eager-only path works without flash-attn present.
"""

from __future__ import annotations

import subprocess
from typing import Any, Optional


def capture_environment() -> dict[str, Any]:
    """Capture runtime environment details for reproducibility metadata.

    Returns
    -------
    dict with keys:
        transformers_version, torch_version, flash_attn_version (may be None),
        cuda_version, gpu_name, gpu_memory_mb, commit_sha
    """
    import torch
    import transformers

    env: dict[str, Any] = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "flash_attn_version": _get_flash_attn_version(),
        "cuda_version": _get_cuda_version(),
        "gpu_name": _get_gpu_name(),
        "gpu_memory_mb": _get_gpu_memory_mb(),
        "commit_sha": _get_git_commit_sha(),
    }
    return env


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_flash_attn_version() -> Optional[str]:
    """Return flash-attn version string, or None if not installed.

    Uses a guarded import — ``flash_attn`` is never loaded if absent.
    """
    try:
        import flash_attn  # type: ignore[import-untyped]

        return getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        return None


def _get_cuda_version() -> Optional[str]:
    """Return CUDA runtime version via torch, or None if unavailable."""
    import torch

    if torch.cuda.is_available():
        return torch.version.cuda
    return None


def _get_gpu_name() -> Optional[str]:
    """Return the name of GPU 0, or None."""
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.get_device_name(0)
    return None


def _get_gpu_memory_mb() -> Optional[int]:
    """Return total memory of GPU 0 in MiB, or None."""
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory // (1024 * 1024)
    return None


def _get_git_commit_sha() -> Optional[str]:
    """Return the current git HEAD SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
