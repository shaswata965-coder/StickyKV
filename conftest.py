"""Pytest configuration for StickyKV.

Registers the ``gpu`` marker so ``-m "not gpu"`` (the CPU-only test invocation
used across the project) runs without an ``PytestUnknownMarkWarning``. GPU tests
are additionally self-skipping when CUDA / triton is unavailable.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: test requires a CUDA GPU (and, for the Triton kernel, triton).",
    )
