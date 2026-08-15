"""Parquet serialization convenience for preprocessing-owned cache keys."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from relarena.cache import CacheConfig, cached_artifact


def cached_frame(
    cache: CacheConfig,
    key: str | Callable[[], str],
    compute: Callable[[], pd.DataFrame],
    *,
    validate: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Load/build one caller-keyed parquet frame through the generic cache.

    A callable `key` is evaluated only for a configured persistent store. This
    avoids expensive content fingerprints when private computation cannot reuse
    an artifact.
    """
    if cache.directory is None or cache.on_miss == "compute":
        frame = compute()
        if validate is not None:
            validate(frame)
        return frame

    resolved_key = key() if callable(key) else key

    def build(path: Path) -> pd.DataFrame:
        frame = compute()
        frame.to_parquet(path)
        return frame

    return cached_artifact(
        cache,
        resolved_key,
        storage="file",
        load=pd.read_parquet,
        build=build,
        validate=validate,
        warm_hint="Run the shared DFS warmer before a configured benchmark.",
    )
