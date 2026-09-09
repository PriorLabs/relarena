"""Optional, experimental helpers for caller-owned local cache artifacts.

This module provides only cache configuration and safe filesystem mechanics. It
does not know what an artifact represents, how its key is constructed, or how it
is serialized. Each preprocessing module owns those decisions and supplies
`load`, `build`, and optional `validate` callbacks to `cached_artifact`.

Using this API is not required: a model may implement caching independently or
avoid persistent caching altogether. Cache warming remains separate from a timed
RelArena experiment. Regardless of implementation, benchmark caches should have
public, reproducible, and leakage-safe generation code; see
`docs/adding-a-model.md` for the contributor policy and reference implementations.

A `CacheConfig` contains one resolved local cache root and a miss policy:

* `raise`: load an existing artifact; raise `CacheMiss` when absent.
* `fill`: load an existing artifact, or build and atomically publish it.
* `compute`: build privately without reading or writing the persistent store.

When no cache directory is configured, artifacts are also built in private
temporary scratch, regardless of the miss policy. Thus `raise` only guards
misses in a configured store.

Artifact keys are opaque, safe relative POSIX paths beneath the cache root.
Artifacts may be files or directories. On a fill miss, construction happens at
a unique staging path; the completed and validated artifact is then published
atomically. Concurrent builders cannot expose partial artifacts. Cache hits are
read-only and create no locks, metadata, or temporary files.

Callers are responsible for:

* constructing keys from the artifact's actual dependencies;
* versioning preprocessing algorithms;
* serialization, loading, building, and semantic validation;
* returning fully materialized values that do not depend on temporary paths.

Environment variables are resolved once by `resolve_cache_config` at an
entrypoint. Models and preprocessing code should receive the resulting immutable
`CacheConfig` explicitly rather than consulting the environment themselves.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar

CACHE_DIR_ENV = "RELARENA_CACHE_DIR"
DISABLE_CACHE_ENV = "RELARENA_DISABLE_CACHE"
LEGACY_DISABLE_CACHE_ENV = "RELARENA_DISABLE_FEATURE_CACHE"

OnMiss = Literal["raise", "fill", "compute"]
Storage = Literal["file", "directory"]
T = TypeVar("T")


class CacheMiss(RuntimeError):
    """A required artifact is absent from a configured cache."""


@dataclass(frozen=True)
class CacheConfig:
    """Resolved cache directory and behavior for an absent artifact."""

    directory: Path | None
    on_miss: OnMiss

    def __post_init__(self) -> None:
        """Reject policies outside the intentionally small public set."""
        if self.on_miss not in ("raise", "fill", "compute"):
            raise ValueError(f"invalid cache miss policy: {self.on_miss!r}")


def parse_local_cache_dir(value: str | Path) -> Path:
    """Expand a local cache path and reject URI-like locations."""
    raw = str(value)
    if not raw:
        raise ValueError("cache directory cannot be empty")
    scheme = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):(?:/{1,2})", raw)
    if scheme is not None and len(scheme.group(1)) > 1:
        raise ValueError(f"cache directory must be a local path, not {raw!r}")
    return Path(raw).expanduser()


def resolve_cache_config(
    cache_dir: str | Path | None, *, on_miss: OnMiss
) -> CacheConfig:
    """Resolve explicit path, environment fallback, and the disable switch once."""
    value = cache_dir
    if value is None:
        value = os.environ.get(CACHE_DIR_ENV) or None
    directory = None if value is None else parse_local_cache_dir(value)
    if os.environ.get(LEGACY_DISABLE_CACHE_ENV):
        warnings.warn(
            f"{LEGACY_DISABLE_CACHE_ENV} is deprecated; use {DISABLE_CACHE_ENV}",
            FutureWarning,
            stacklevel=2,
        )
        directory = None
    if os.environ.get(DISABLE_CACHE_ENV):
        directory = None
    return CacheConfig(directory=directory, on_miss=on_miss)


def cache_key(*segments: str) -> PurePosixPath:
    """Join validated path segments into a canonical relative cache key."""
    if not segments:
        raise ValueError("cache key needs at least one segment")
    for segment in segments:
        if not segment or segment in (".", ".."):
            raise ValueError(f"invalid cache-key segment: {segment!r}")
        if "/" in segment or "\\" in segment:
            raise ValueError(f"cache-key segment contains a separator: {segment!r}")
    return PurePosixPath(*segments)


def _key_parts(key: str | PurePosixPath) -> tuple[str, ...]:
    raw = str(key)
    if not raw or raw.startswith("/") or "\\" in raw:
        raise ValueError(f"cache key must be a safe relative POSIX path: {raw!r}")
    parts = tuple(raw.split("/"))
    if any(not part or part in (".", "..") for part in parts):
        raise ValueError(f"cache key must be a safe relative POSIX path: {raw!r}")
    return parts


def _destination(root: Path, parts: tuple[str, ...]) -> Path:
    resolved_root = root.resolve()
    destination = root.joinpath(*parts)
    if not destination.resolve().is_relative_to(resolved_root):
        raise ValueError("cache key escapes the configured cache directory")
    return destination


def _qualifies(path: Path, storage: Storage) -> bool:
    return path.is_file() if storage == "file" else path.is_dir()


def _require_shape(path: Path, storage: Storage) -> None:
    if not _qualifies(path, storage):
        raise RuntimeError(f"cache builder did not create {storage} artifact at {path}")


def _discard(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _build_in_scratch(
    parts: tuple[str, ...],
    storage: Storage,
    build: Callable[[Path], T],
    validate: Callable[[T], None] | None,
) -> T:
    with tempfile.TemporaryDirectory(prefix="relarena-cache-") as scratch:
        path = Path(scratch).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = build(path)
        _require_shape(path, storage)
        if validate is not None:
            validate(value)
        return value


def cached_artifact(
    cache: CacheConfig,
    key: str | PurePosixPath,
    *,
    storage: Storage,
    load: Callable[[Path], T],
    build: Callable[[Path], T],
    validate: Callable[[T], None] | None = None,
    warm_hint: str | None = None,
) -> T:
    """Load, build, or reject one opaque file or directory artifact."""
    if storage not in ("file", "directory"):
        raise ValueError(f"invalid cache storage shape: {storage!r}")
    parts = _key_parts(key)
    if cache.directory is None or cache.on_miss == "compute":
        return _build_in_scratch(parts, storage, build, validate)

    destination = _destination(cache.directory, parts)
    if _qualifies(destination, storage):
        value = load(destination)
        if validate is not None:
            validate(value)
        return value

    if cache.on_miss == "raise":
        hint = "" if warm_hint is None else f" {warm_hint}"
        raise CacheMiss(
            f"cache miss for {str(key)!r} in {cache.directory}.{hint}".rstrip()
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        value = build(staging)
        _require_shape(staging, storage)
        if validate is not None:
            validate(value)
        if storage == "file":
            os.replace(staging, destination)
        else:
            try:
                staging.rename(destination)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not _qualifies(
                    destination, storage
                ):
                    raise
        return value
    finally:
        _discard(staging)
