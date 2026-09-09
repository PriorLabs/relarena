"""A novel model-owned cache with formats unknown to relarena.cache."""

from __future__ import annotations

import hashlib
from pathlib import Path

from relarena.cache import CacheConfig, cache_key, cached_artifact

_VERSION = 7


def _key(raw: bytes, name: str) -> str:
    fingerprint = hashlib.blake2s(raw, digest_size=8).hexdigest()
    return str(
        cache_key("fixture-model", f"preprocessing-v{_VERSION}", fingerprint, name)
    )


def novel_binary(raw: bytes, cache: CacheConfig) -> bytes:
    """Own a made-up single-file codec without registering it centrally."""
    return cached_artifact(
        cache,
        _key(raw, "features.bin"),
        storage="file",
        load=Path.read_bytes,
        build=lambda path: _write_binary(path, raw[::-1]),
        validate=lambda value: _validate(value, raw[::-1]),
        warm_hint="Run python -m fixture_model.warm_cache.",
    )


def novel_directory(raw: bytes, cache: CacheConfig) -> bytes:
    """Own a made-up multi-file directory codec and its validation."""
    return cached_artifact(
        cache,
        _key(raw, "index.novel"),
        storage="directory",
        load=lambda path: (path / "payload.bin").read_bytes(),
        build=lambda path: _write_directory(path, raw.upper()),
        validate=lambda value: _validate(value, raw.upper()),
    )


def warm_cache(raw: bytes, cache_dir: Path) -> None:
    """Public-warmer shape a contributor can expose from its own module."""
    cache = CacheConfig(cache_dir, "fill")
    novel_binary(raw, cache)
    novel_directory(raw, cache)


def _write_binary(path: Path, value: bytes) -> bytes:
    path.write_bytes(value)
    return value


def _write_directory(path: Path, value: bytes) -> bytes:
    path.mkdir()
    (path / "payload.bin").write_bytes(value)
    (path / "format-version").write_text("1")
    return value


def _validate(value: bytes, expected: bytes) -> None:
    if value != expected:
        raise ValueError("invalid fixture artifact")
