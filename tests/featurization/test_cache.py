"""Tests for the DFS module's parquet cache convenience wrapper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from relarena.cache import CacheConfig, CacheMiss
from relarena.featurization.cache import cached_frame


def _frame(value: int = 1) -> pd.DataFrame:
    return pd.DataFrame({"value": [value]})


def test__cached_frame__fill_then_read_only_hit__round_trips(tmp_path: Path) -> None:
    key = "dfs/v-test/example/matrix.parquet"
    filled = cached_frame(CacheConfig(tmp_path, "fill"), key, _frame)
    hit = cached_frame(
        CacheConfig(tmp_path, "raise"),
        key,
        lambda: pytest.fail("hit must not compute"),
    )
    pd.testing.assert_frame_equal(filled, hit)


def test__cached_frame__configured_miss__uses_dfs_warm_hint(
    tmp_path: Path,
) -> None:
    with pytest.raises(CacheMiss, match="shared DFS warmer"):
        cached_frame(
            CacheConfig(tmp_path, "raise"),
            "dfs/v-test/example/matrix.parquet",
            _frame,
        )


def test__cached_frame__validation__runs_on_fill_and_hit(tmp_path: Path) -> None:
    seen: list[int] = []

    def validate(frame: pd.DataFrame) -> None:
        seen.append(int(frame.iloc[0, 0]))

    key = "dfs/v-test/example/matrix.parquet"
    cached_frame(CacheConfig(tmp_path, "fill"), key, _frame, validate=validate)
    cached_frame(CacheConfig(tmp_path, "raise"), key, _frame, validate=validate)
    assert seen == [1, 1]


@pytest.mark.parametrize("configured", [False, True])
def test__cached_frame__private_compute__skips_parquet_round_trip(
    configured: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda *args, **kwargs: pytest.fail("private compute must not serialize"),
    )
    cache = CacheConfig(tmp_path if configured else None, "compute")

    frame = cached_frame(
        cache,
        lambda: pytest.fail("private compute must not construct a cache key"),
        _frame,
    )

    pd.testing.assert_frame_equal(frame, _frame())


def test__cached_frame__configured_store__resolves_lazy_key(tmp_path: Path) -> None:
    frame = cached_frame(
        CacheConfig(tmp_path, "fill"),
        lambda: "dfs/v-test/example/matrix.parquet",
        _frame,
    )

    pd.testing.assert_frame_equal(frame, _frame())
    assert (tmp_path / "dfs/v-test/example/matrix.parquet").is_file()
