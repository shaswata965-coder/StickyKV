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

from typing import Any, Callable, Optional, Tuple, Type

from utils.config import ConfigValidationError

__all__ = [
    "ConfigValidationError",
    "get_cache_classes",
    "validate_backend_attn_pairing",
    "assert_transformers_version_supported",
    "is_transformers_version_supported",
    "MAX_SUPPORTED_TRANSFORMERS",
]


_BACKEND_TO_ATTN_IMPL = {
    "flash_attn": ("flash_attention_2",),
    "eager": ("eager",),
}

# Highest transformers version the windowed KV cache is known-correct on.
# RoPE positioning is no longer version-coupled: every eviction compacts AND
# re-rotates surviving keys to contiguous positions, and a forward pre-hook
# (utils.position_override) overrides the query's position to the compacted
# cache length each step — set explicitly, so it does not depend on how HF
# derives ``cache_position``. The remaining reason to pin <= 4.47.1 is a
# SEPARATE incompatibility: transformers 5.x builds the causal mask via
# ``create_causal_mask`` → ``Cache.get_mask_sizes()``, which ``WindowedCache``
# does not implement, so a full-model forward crashes on 5.x (see the
# "Environment caveat" in design.md). Until that mask-API gap is closed, refuse
# to run on a newer version rather than crash mid-run.
MAX_SUPPORTED_TRANSFORMERS: Tuple[int, int, int] = (4, 47, 1)


def _parse_version(version: str) -> Tuple[int, int, int]:
    """Parse a version string to a ``(major, minor, patch)`` int tuple.

    Tolerant of suffixes like ``4.47.1.dev0`` / ``4.47.1+cu121`` — only the
    leading numeric ``major.minor.patch`` is used.
    """
    parts = []
    for chunk in version.split(".")[:3]:
        # Take only the LEADING run of digits, so "1+cu121" → 1 and
        # "1rc2" → 1 (not "1121" / "12").
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def is_transformers_version_supported(
    version: str,
    max_version: Tuple[int, int, int] = MAX_SUPPORTED_TRANSFORMERS,
) -> bool:
    """Return ``True`` iff *version* is <= *max_version* (inclusive)."""
    return _parse_version(version) <= max_version


def assert_transformers_version_supported(version: Optional[str] = None) -> None:
    """Raise ``ConfigValidationError`` if transformers is newer than supported.

    Called at the start of any *real* windowed-cache run (model-backed). Not
    invoked from pure-logic unit tests, which must stay runnable on dev boxes
    that have a newer transformers installed.

    Parameters
    ----------
    version : str, optional
        Version string to check. Defaults to the installed
        ``transformers.__version__``.
    """
    if version is None:
        import transformers

        version = transformers.__version__

    if not is_transformers_version_supported(version):
        supported = ".".join(str(p) for p in MAX_SUPPORTED_TRANSFORMERS)
        raise ConfigValidationError(
            f"transformers {version} is newer than the supported {supported}. "
            "RoPE positioning is version-independent (the cache compacts + "
            "re-rotates every eviction and a pre-hook overrides the query "
            "position explicitly), but transformers 5.x builds the causal mask "
            "via create_causal_mask -> Cache.get_mask_sizes(), which "
            "WindowedCache does not implement, so a full-model forward crashes. "
            f"Pin transformers=={supported} (see environment.yml), or implement "
            "the Cache mask API and raise MAX_SUPPORTED_TRANSFORMERS before bumping."
        )


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
