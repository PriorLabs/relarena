"""Tests for the `relarena` CLI result assembly.

The CLI shares its results schema with batch sweeps by going through
`results.summary_to_dataframe`; these tests pin that the CSV it writes
carries every native metric (not just the primary) and the full identity
columns. `run_experiment` and task discovery are stubbed so nothing downloads
or trains.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from relbench.base import TaskType

from relarena import cli
from relarena.results import SystemResult, TrialResult
from relarena.runner import ExperimentSummary, SystemExperimentSummary
from relarena.tasks import TaskSpec


def _summary() -> ExperimentSummary:
    default = TrialResult(
        config={"a": 1},
        config_id="d",
        config_tag="default",
        val_score=0.8,
        test_score=None,
        val_metrics={"roc_auc": 0.8, "f1": 0.7},
        test_metrics={},
    )
    best = TrialResult(
        config={"a": 2},
        config_id="b",
        config_tag="r0",
        val_score=0.9,
        test_score=0.88,
        val_metrics={"roc_auc": 0.9, "f1": 0.75},
        test_metrics={"roc_auc": 0.88, "f1": 0.7},
    )
    return ExperimentSummary(
        model_name="constant-global",
        dataset="rel-f1",
        task_name="driver-dnf",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_name="roc_auc",
        seed=0,
        n_trials=10,
        default=default,
        tuned=best,
        trials=[default, best],
        peak_rss_gib=1.25,
    )


def test__cli__successful_runs__writes_all_metrics_for_every_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = TaskSpec("rel-f1", "driver-dnf", TaskType.BINARY_CLASSIFICATION)
    monkeypatch.setattr(cli, "list_entity_tasks", lambda datasets: [spec])
    monkeypatch.setattr(cli, "run_experiment", lambda *a, **k: _summary())

    out = tmp_path / "results.csv"
    assert cli.main(["--model", "constant-global", "--output", str(out)]) == 0

    df = pd.read_csv(out)
    assert len(df) == 2
    assert {
        "model",
        "dataset",
        "task",
        "task_type",
        "seed",
        "n_trials",
        "metric",
        "selected",
        "val_roc_auc",
        "val_f1",
        "test_roc_auc",
        "test_f1",
        "peak_rss_gib",
    }.issubset(df.columns)
    assert "id" not in df.columns

    selected = df[df["selected"]]
    assert len(selected) == 1
    assert selected["config_id"].iloc[0] == "b"
    assert selected["test_roc_auc"].iloc[0] == pytest.approx(0.88)
    assert selected["peak_rss_gib"].iloc[0] == pytest.approx(1.25)


def test__cli__no_successful_runs__returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = TaskSpec("rel-f1", "driver-dnf", TaskType.BINARY_CLASSIFICATION)
    monkeypatch.setattr(cli, "list_entity_tasks", lambda datasets: [spec])

    def boom(*a: object, **k: object) -> ExperimentSummary:
        raise RuntimeError("download failed")

    monkeypatch.setattr(cli, "run_experiment", boom)
    assert cli.main(["--model", "constant-global"]) == 1


def test__cli__cache_dir__is_forwarded_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = TaskSpec("rel-f1", "driver-dnf", TaskType.BINARY_CLASSIFICATION)
    monkeypatch.setattr(cli, "list_entity_tasks", lambda datasets: [spec])
    seen: dict[str, object] = {}

    def run(*args: object, **kwargs: object) -> ExperimentSummary:
        seen.update(kwargs)
        return _summary()

    monkeypatch.setattr(cli, "run_experiment", run)
    assert cli.main(["--model", "constant-global", "--cache-dir", str(tmp_path)]) == 0
    assert seen["cache_dir"] == str(tmp_path)


def test__cli__parallel_tasks__runs_tasks_in_workers_and_preserves_output_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = [
        TaskSpec("rel-f1", "second", TaskType.BINARY_CLASSIFICATION),
        TaskSpec("rel-f1", "first", TaskType.BINARY_CLASSIFICATION),
    ]
    monkeypatch.setattr(cli, "list_entity_tasks", lambda datasets: specs)
    seen_workers: list[int] = []
    seen_calls: list[tuple[str, str, dict[str, object]]] = []

    def run(
        model_cls: object,
        dataset: str,
        task: str,
        **kwargs: object,
    ) -> ExperimentSummary:
        seen_calls.append((dataset, task, kwargs))
        return replace(_summary(), dataset=dataset, task_name=task)

    class ImmediateExecutor:
        def __init__(self, *, max_workers: int, max_tasks_per_child: int) -> None:
            seen_workers.append(max_workers)
            assert max_tasks_per_child == 1

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future:
            future: Future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - mirrors executor contract
                future.set_exception(exc)
            return future

    monkeypatch.setattr(cli, "run_experiment", run)
    monkeypatch.setattr(cli, "ProcessPoolExecutor", ImmediateExecutor)
    out = tmp_path / "parallel.csv"

    assert (
        cli.main(
            [
                "--model",
                "constant-global",
                "--parallel-tasks",
                "2",
                "--output",
                str(out),
            ]
        )
        == 0
    )

    assert seen_workers == [2]
    assert [task for _, task, _ in seen_calls] == ["second", "first"]
    assert all(call[2]["cache_predictions"] is False for call in seen_calls)
    selected = pd.read_csv(out).query("selected")
    assert selected["task"].tolist() == ["second", "first"]


def test__cli__parallel_tasks__must_be_positive() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--model", "constant-global", "--parallel-tasks", "0"])


def test__cli__system_summary__writes_native_system_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = TaskSpec("rel-f1", "driver-dnf", TaskType.BINARY_CLASSIFICATION)
    monkeypatch.setattr(cli, "list_entity_tasks", lambda datasets: [spec])
    summary = SystemExperimentSummary(
        system_name="rt-plurel",
        dataset="rel-f1",
        task_name="driver-dnf",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_name="roc_auc",
        seed=0,
        result=SystemResult(
            test_score=0.9,
            test_metrics={"roc_auc": 0.9},
            time_total=12.0,
        ),
        peak_rss_gib=2.5,
    )
    monkeypatch.setattr(cli, "run_experiment", lambda *a, **k: summary)

    out = tmp_path / "results.csv"
    assert cli.main(["--model", "rt-plurel", "--output", str(out)]) == 0

    row = pd.read_csv(out).iloc[0]
    assert row["kind"] == "system"
    assert row["time_total"] == pytest.approx(12.0)
    assert row["peak_rss_gib"] == pytest.approx(2.5)
    assert row["selected"]
    assert "config" not in row.index
    assert "val_score" not in row.index
