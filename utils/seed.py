"""Deterministic seeding for Python, NumPy, and PyTorch RNGs.

Usage:
    from utils.seed import seed_everything, SeedContext

    # Global seeding
    seed_everything(42)

    # Scoped seeding in tests
    with SeedContext(42):
        ...
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all RNGs to *seed* and enable deterministic CUDA algorithms.

    Sets:
    - ``PYTHONHASHSEED`` env var
    - ``random`` stdlib
    - ``numpy`` global RNG
    - ``torch`` CPU + CUDA RNGs
    - ``torch.use_deterministic_algorithms(True)``
    - ``torch.backends.cudnn.{deterministic, benchmark}``
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    # NOTE: Setting PYTHONHASHSEED here does not affect this process's hash
    # randomization (that's fixed at interpreter startup); it only propagates
    # to subprocesses launched after this point.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # warn_only=True keeps the deterministic intent without crashing on ops
    # that lack a deterministic implementation (some scatter/index_put paths).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch without warn_only kwarg.
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


@contextmanager
def SeedContext(seed: int) -> Generator[None, None, None]:
    """Context manager that seeds all RNGs on entry and restores state on exit.

    Useful for tests that need scoped determinism without polluting the
    global state of other tests.
    """
    # Capture pre-existing state
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    old_det = torch.are_deterministic_algorithms_enabled()
    old_hash = os.environ.get("PYTHONHASHSEED")

    try:
        seed_everything(seed)
        yield
    finally:
        # Restore state
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available():
            for i, s in enumerate(cuda_states):
                torch.cuda.set_rng_state(s, i)
        torch.use_deterministic_algorithms(old_det)
        if old_hash is not None:
            os.environ["PYTHONHASHSEED"] = old_hash
        elif "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]
