"""SHA-256 hashing utilities for integrity checks and article identification.

Provides:
- ``sha256_file(path)`` — full hex digest of a file (for npz integrity).
- ``sha256_string(text)`` — 16-char truncated hex digest (for article SHAs).
- ``sha256_tokenizer(tokenizer)`` — digest of tokenizer vocab for drift detection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the full SHA-256 hex digest of *path*.

    Reads in 64 KiB chunks to handle large files without excessive memory.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_string(text: str) -> str:
    """Return a **16-character** truncated SHA-256 hex digest of *text*.

    This is the canonical format for article IDs in the telemetry schema.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sha256_tokenizer(tokenizer: Any) -> str:
    """Return a SHA-256 hex digest capturing the tokenizer vocabulary state.

    The digest is computed over the sorted ``(token, id)`` pairs serialized
    as JSON.  This detects vocabulary drift across ``transformers`` versions
    without depending on internal tokenizer serialization details.

    Parameters
    ----------
    tokenizer
        Any HuggingFace tokenizer exposing a ``get_vocab()`` method.
    """
    vocab = tokenizer.get_vocab()
    # Canonical ordering: sort by token string to ensure determinism
    canonical = json.dumps(sorted(vocab.items()), ensure_ascii=True, sort_keys=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
