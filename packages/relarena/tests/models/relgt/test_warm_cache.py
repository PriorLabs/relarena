"""Tests for RelGT's directly runnable token warmer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.models.relgt import warm_cache


def test__precompute_dataset_task__fills_every_phase_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inner = SimpleNamespace(db_state="idb", train_table="it", eval_table="iv")
    outer = SimpleNamespace(
        db_state="odb", train_table="ot", eval_table="oe", val_table="ov"
    )
    source = SimpleNamespace(
        task="task",
        inner_split=lambda: inner,
        outer_split=lambda: outer,
        run_identity=lambda phase, **kwargs: RunIdentity(
            "dataset", "db", "task", "labels", phase=phase
        ),
    )
    monkeypatch.setattr(warm_cache, "RelBenchDatasetTask", lambda *a, **k: source)
    monkeypatch.setattr(warm_cache, "build_graph", lambda db, device: (f"g-{db}", {}))
    calls: list[tuple] = []
    monkeypatch.setattr(
        warm_cache,
        "_load_precompute_tokens",
        lambda: lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = warm_cache.precompute_dataset_task(
        "dataset", "task", cache_dir=tmp_path, download=False, num_workers=2
    )

    assert result == "dataset/task"
    assert [call[0][2] for call in calls] == [["it", "iv"], ["ot", "oe", "ov"]]
    assert all(call[0][4] == CacheConfig(tmp_path, "fill") for call in calls)
    assert [call[1]["run_identity"].phase for call in calls] == ["inner", "outer"]
    assert all(call[1]["num_workers"] == 2 for call in calls)
