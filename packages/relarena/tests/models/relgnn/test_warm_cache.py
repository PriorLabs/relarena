"""Tests for RelGNN's directly runnable graph warmer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.models.relgnn import warm_cache


def test__precompute_dataset_task__fills_inner_and_outer_graphs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = SimpleNamespace(
        inner_split=lambda: SimpleNamespace(db_state="inner-db"),
        outer_split=lambda: SimpleNamespace(db_state="outer-db"),
        run_identity=lambda phase, **kwargs: RunIdentity(
            "dataset", "db", None, None, phase=phase
        ),
    )
    monkeypatch.setattr(warm_cache, "RelBenchDatasetTask", lambda *a, **k: source)
    calls: list[tuple[object, CacheConfig, RunIdentity]] = []
    monkeypatch.setattr(
        warm_cache,
        "load_graph",
        lambda db, cache, identity: calls.append((db, cache, identity)),
    )

    result = warm_cache.precompute_dataset_task(
        "dataset", "representative-task", cache_dir=tmp_path, download=False
    )

    assert result == "dataset/representative-task"
    assert [call[0] for call in calls] == ["inner-db", "outer-db"]
    assert [call[2].phase for call in calls] == ["inner", "outer"]
    assert all(call[1] == CacheConfig(tmp_path, "fill") for call in calls)
