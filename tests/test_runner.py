"""Tests for the experiment runner (`relarena.runner`).

The download, splits, and tuner are stubbed so nothing hits RelBench.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from relbench.base import TaskType
from relbench.metrics import roc_auc

from relarena import runner
from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.results import TrialResult
from relarena.system import RelArenaSystem

_MODEL = SimpleNamespace(
    name="stub", supported_task_types=frozenset({TaskType.BINARY_CLASSIFICATION})
)
_IDENTITY = RunIdentity("rel-f1", "db", "driver-dnf", "task")


def _trial(
    tag: str, error: str | None = None, *, val_score: float = 0.9
) -> TrialResult:
    return TrialResult(
        config={"tag": tag},
        config_id=tag,
        config_tag=tag,
        val_score=None if error else val_score,
        error=error,
    )


@pytest.mark.parametrize("nonfinite_score", [math.nan, math.inf, -math.inf])
def test__select_best__nonfinite_incumbent__selects_finite_trial(
    nonfinite_score: float,
) -> None:
    trials = [
        _trial("nonfinite", val_score=nonfinite_score),
        _trial("finite", val_score=0.8),
    ]

    assert runner.select_best(trials, roc_auc).config_tag == "finite"


def test__select_best__only_nonfinite_scores__raises() -> None:
    trials = [
        _trial("nan", val_score=math.nan),
        _trial("positive-infinity", val_score=math.inf),
        _trial("negative-infinity", val_score=-math.inf),
    ]

    with pytest.raises(RuntimeError, match="finite validation score"):
        runner.select_best(trials, roc_auc)


@pytest.fixture
def refit_tags(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub a run whose second of two trials errored; collect the refit configs."""
    monkeypatch.setattr(
        runner,
        "RelBenchDatasetTask",
        lambda *a, **k: SimpleNamespace(
            task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
            metric=SimpleNamespace(__name__="roc_auc"),
            inner_split=lambda: None,
            outer_split=lambda: None,
            run_identity=lambda phase, **kwargs: _IDENTITY.for_phase(phase),
        ),
    )
    monkeypatch.setattr(
        runner, "tune", lambda *a, **k: [_trial("default"), _trial("r1", "overloaded")]
    )
    tags: list[str] = []

    def refit(model_cls: Any, config: dict, *a: Any, **k: Any) -> dict:
        tags.append(config["tag"])
        return {
            "test_score": 0.75,
            "test_metrics": {},
            "test_pred": None,
            "fit_time_refit": 1.0,
            "predict_time_refit": 0.5,
        }

    monkeypatch.setattr(runner, "refit_and_evaluate", refit)
    return tags


def _run(**kwargs: Any) -> Any:
    return runner.run_model_experiment(
        _MODEL, "rel-f1", "driver-dnf", search_space=object(), download=False, **kwargs
    )


def test__run_model_experiment__a_trial_errored__raises_without_refitting(
    refit_tags: list[str],
) -> None:
    with pytest.raises(RuntimeError, match="1 of 2 trials failed"):
        _run()
    assert refit_tags == []  # the outer experiment never started


def test__run_model_experiment__require_all_trials_off__refits_the_surviving_config(
    refit_tags: list[str],
) -> None:
    summary = _run(require_all_trials=False)

    assert refit_tags == ["default"]
    assert summary.tuned is not None and summary.tuned.test_score == 0.75


def test__run_model_experiment__cache_dir__passes_one_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        metric=SimpleNamespace(__name__="roc_auc"),
        inner_split=lambda: None,
        run_identity=lambda phase, **kwargs: _IDENTITY.for_phase(phase),
    )
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *args, **kwargs: source)
    seen: list[tuple[CacheConfig, RunIdentity]] = []

    def tune(*args: Any, **kwargs: Any) -> list[TrialResult]:
        seen.append((kwargs["cache"], kwargs["run_identity"]))
        return [_trial("default")]

    monkeypatch.setattr(runner, "tune", tune)
    runner.run_model_experiment(
        _MODEL,
        "rel-f1",
        "driver-dnf",
        search_space=object(),
        download=False,
        evaluate_test=False,
        cache_dir=tmp_path,
    )
    assert seen == [(CacheConfig(tmp_path, "raise"), _IDENTITY.for_phase("inner"))]


def test__run_experiment__model_dispatches_to_model_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SimpleNamespace(peak_rss_gib=None)
    seen: dict[str, Any] = {}

    def run_model(*args: Any, **kwargs: Any) -> object:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(runner, "run_model_experiment", run_model)

    actual = runner.run_experiment(
        _MODEL,
        "rel-f1",
        "driver-dnf",
        search_space=object(),
        n_trials=3,
        download=False,
    )

    assert actual is expected
    assert seen["args"] == (_MODEL, "rel-f1", "driver-dnf")
    assert seen["kwargs"]["n_trials"] == 3


def _system_source() -> SimpleNamespace:
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metrics=[roc_auc],
        evaluate=lambda pred, target, metrics: {"roc_auc": 0.75},
    )
    outer = SimpleNamespace(eval_table=SimpleNamespace(df=[{}, {}, {}]))
    return SimpleNamespace(
        task=task,
        metric=roc_auc,
        inner_split=lambda: "inner",
        outer_split=lambda: outer,
        run_identity=lambda phase=None: _IDENTITY.for_phase(phase),
    )


def test__run_system_experiment__passes_both_splits_and_records_one_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _system_source()
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: source)
    seen: dict[str, Any] = {}

    class S(RelArenaSystem):
        name = "system"

        def run(self, task: Any, **kwargs: Any) -> np.ndarray:
            seen.update(kwargs)
            seen["identity"] = self.run_identity
            return np.array([0.1, 0.2, 0.3])

    summary = runner.run_system_experiment(
        S, "rel-f1", "driver-dnf", seed=4, download=False
    )

    assert seen["inner_split"] == "inner"
    assert seen["outer_split"] is source.outer_split()
    assert seen["seed"] == 4
    assert seen["identity"] == _IDENTITY
    assert summary.result.test_score == pytest.approx(0.75)
    assert summary.result.test_pred is not None
    assert summary.result.test_pred.tolist() == [0.1, 0.2, 0.3]


def test__run_experiment__native_system_dispatches_without_search_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _system_source()
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: source)

    class S(RelArenaSystem):
        name = "system"

        def run(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return np.zeros(3)

    summary = runner.run_experiment(S, "rel-f1", "driver-dnf", download=False)

    assert isinstance(summary, runner.SystemExperimentSummary)


def test__run_system_experiment__wrong_prediction_shape__raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: _system_source())

    class S(RelArenaSystem):
        name = "system"

        def run(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return np.zeros((3, 1))

    with pytest.raises(ValueError, match=r"expected \(3,\)"):
        runner.run_system_experiment(S, "rel-f1", "driver-dnf", download=False)


def test__peak_rss_monitor__resets_between_sequential_tasks() -> None:
    gib = 1024**3

    class Process:
        def __init__(self, samples: list[int]) -> None:
            self.samples = iter(samples)

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=next(self.samples))

    first = runner._PeakRSSMonitor(
        process=Process([1 * gib, 3 * gib, 2 * gib]),
        interval_seconds=60,
    )
    with first:
        first._sample()

    second = runner._PeakRSSMonitor(
        process=Process([1 * gib, 2 * gib, 1 * gib]),
        interval_seconds=60,
    )
    with second:
        second._sample()

    assert first.peak_rss_gib == pytest.approx(3.0)
    assert second.peak_rss_gib == pytest.approx(2.0)
