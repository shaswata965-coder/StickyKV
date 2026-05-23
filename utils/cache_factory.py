"""Cache factory — single source of truth for backend selection.

Shell only for Prompt 01. The cache packages it routes to are built in
Prompt 02.  Two functions:

- ``get_cache_classes(backend)`` — lazy-imports and returns the
  ``(WindowedCache, WindowedCacheConfig, install_score_hooks)`` trio for
  either ``'flash_attn'`` or ``'eager'``.
- ``validate_backend_attn_pairing(backend, attn_implementation)`` — raises
  ``ConfigValidationError`` on mismatched backend/attention pairs.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Type

from utils.config import ConfigValidationError

__all__ = ["ConfigValidationError", "get_cache_classes", "validate_backend_attn_pairing"]


_BACKEND_TO_ATTN_IMPL = {
    "flash_attn": ("flash_attention_2",),
    "eager": ("eager",),
}


def get_cache_classes(backend: str) -> Tuple[Type, Type, Callable]:
    """Return ``(WindowedCache, WindowedCacheConfig, install_score_hooks)``
    for the requested backend.

    Parameters
    ----------
    backend : str
        Either ``'flash_attn'`` or ``'eager'``.

    Returns
    -------
    tuple[type, type, callable]
        The cache class, its config class, and the hook installer.

    Raises
    ------
    ValueError
        If *backend* is not recognized.
    ConfigValidationError
        If the required package is not installed (flash_attn only).

    Notes
    -----
    Imports are lazy: ``flash_attn`` is **never** imported on the eager
    path, so the eager backend runs without flash-attn installed.
    """
    if backend == "flash_attn":
        try:
            from modules.windowed_cache import (  # type: ignore[attr-defined]
                WindowedCache,
                WindowedCacheConfig,
                install_score_hooks,
            )
        except ImportError as e:
            raise ConfigValidationError(
                "flash_attn backend requested but modules.windowed_cache is not "
                "available.  Ensure Prompt 02 has been implemented."
            ) from e
        return WindowedCache, WindowedCacheConfig, install_score_hooks

    elif backend == "eager":
        try:
            from modules.windowed_eager_cache import (  # type: ignore[attr-defined]
                WindowedCache,
                WindowedCacheConfig,
                install_score_hooks,
            )
        except ImportError as e:
            raise ConfigValidationError(
                "eager backend requested but modules.windowed_eager_cache is not "
                "available.  Ensure Prompt 02 has been implemented."
            ) from e
        return WindowedCache, WindowedCacheConfig, install_score_hooks

    else:
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}.  Must be 'flash_attn' or 'eager'."
        )


def validate_backend_attn_pairing(
    backend: str, attn_implementation: str
) -> None:
    """Validate that the cache backend and attention implementation are compatible.

    Rules:
    - ``'flash_attn'`` backend requires ``'flash_attention_2'`` attention.
    - ``'eager'`` backend requires ``'eager'`` attention.

    Called before model load so a mismatched config fails fast.

    Raises
    ------
    ConfigValidationError
        On mismatch.
    """
    if backend not in _BACKEND_TO_ATTN_IMPL:
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}.  Must be 'flash_attn' or 'eager'."
        )

    allowed = _BACKEND_TO_ATTN_IMPL[backend]
    if attn_implementation not in allowed:
        raise ConfigValidationError(
            f"Cache backend {backend!r} requires "
            f"attn_implementation in {allowed!r}, but got "
            f"{attn_implementation!r}.  Fix your config."
        )
