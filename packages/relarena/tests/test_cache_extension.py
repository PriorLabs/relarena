"""Prove a contributor can own novel formats without changing cache plumbing."""

from __future__ import annotations

import shutil
from pathlib import Path

from relarena.cache import CacheConfig
from tests.fixtures.cached_model import novel_binary, novel_directory, warm_cache


def _snapshot(root: Path) -> dict[Path, bytes | None]:
    """Capture the exact relative tree and file contents beneath `root`."""
    return {
        path.relative_to(root): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test__model_owned_cache__unknown_file_and_directory__warm_and_read(
    tmp_path: Path,
) -> None:
    raw = b"model input"
    original = tmp_path / "original"
    warm_cache(raw, original)

    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    before = _snapshot(relocated)
    read_only = CacheConfig(relocated, "raise")

    assert novel_binary(raw, read_only) == raw[::-1]
    assert novel_directory(raw, read_only) == raw.upper()
    assert _snapshot(relocated) == before
