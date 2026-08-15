"""Tests for the directly runnable shared DFS warmer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from relarena.cache import CacheConfig
from relarena.featurization import warm_cache
from relarena.identity import RunIdentity


def test__warm_dfs_cache__uses_one_shared_preprocessor_for_both_phases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inner = SimpleNamespace(db_state="inner-db", train_table="it", eval_table="iv")
    outer = SimpleNamespace(
        db_state="outer-db", train_table="ot", val_table="ov", eval_table="oe"
    )
    source = SimpleNamespace(
        task=SimpleNamespace(time_col="time"),
        inner_split=lambda: inner,
        outer_split=lambda: outer,
        run_identity=lambda phase, **kwargs: RunIdentity(
            "dataset", "db", "task", "labels", phase=phase
        ),
    )
    monkeypatch.setattr(warm_cache, "concat_tables", lambda a, b: "ot+ov")
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        warm_cache,
        "build_dfs_features",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    config = CacheConfig(tmp_path, "fill")

    warm_cache.warm_dfs_cache(source, config, max_depth=3)

    assert [(args[1], args[2], kwargs["history_table"]) for args, kwargs in calls] == [
        ("inner-db", "it", "it"),
        ("inner-db", "iv", "it"),
        ("outer-db", "ot", "ot"),
        ("outer-db", "oe", "ot"),
        ("outer-db", "ot+ov", "ot+ov"),
        ("outer-db", "oe", "ot+ov"),
    ]
    assert {kwargs["run_identity"].phase for _, kwargs in calls} == {
        "inner",
        "outer",
    }
    assert all(kwargs["cache"] is config for _, kwargs in calls)
    assert all(kwargs["max_depth"] == 3 for _, kwargs in calls)


def test__warm_dfs_cache__non_fill_config__raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="on_miss='fill'"):
        warm_cache.warm_dfs_cache(None, CacheConfig(tmp_path, "raise"))
