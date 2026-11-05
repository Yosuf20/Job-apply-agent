"""SHA-256 helpers used for tracker provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
