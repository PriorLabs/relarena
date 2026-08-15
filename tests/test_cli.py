"""Tests for the `relarena` CLI result assembly.

The CLI shares its results schema with batch sweeps by going through
`results.summary_to_dataframe`; these tests pin that the CSV it writes
carries every native metric (not just the primary) and the full identity
columns. `run_experiment` and task discovery are stubbed so nothing downloads
or trains.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from relbench.base import TaskType

from relarena import cli
from relarena.results import TrialResult
from relarena.runner import ExperimentSummary
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
    # All configs are written (default + tuned), with every native metric expanded
    # into val_<m> / test_<m> columns and the full identity, mirroring the sweep.
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
    }.issubset(df.columns)
    assert "id" not in df.columns  # in-process CLI run: no distributed cache id

    selected = df[df["selected"]]
    assert len(selected) == 1
    assert selected["config_id"].iloc[0] == "b"
    assert selected["test_roc_auc"].iloc[0] == pytest.approx(0.88)


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
