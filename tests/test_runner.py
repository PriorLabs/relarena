"""Tests for the experiment runner (`relarena.runner`).

The download, splits, and tuner are stubbed so nothing hits RelBench.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from relbench.base import TaskType

from relarena import runner
from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.results import TrialResult

_MODEL = SimpleNamespace(
    name="stub", supported_task_types=frozenset({TaskType.BINARY_CLASSIFICATION})
)
_IDENTITY = RunIdentity("rel-f1", "db", "driver-dnf", "task")


def _trial(tag: str, error: str | None = None) -> TrialResult:
    return TrialResult(
        config={"tag": tag},
        config_id=tag,
        config_tag=tag,
        val_score=None if error else 0.9,
        error=error,
    )


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
    return runner.run_experiment(
        _MODEL, "rel-f1", "driver-dnf", search_space=object(), download=False, **kwargs
    )


def test__run_experiment__a_trial_errored__raises_without_refitting(
    refit_tags: list[str],
) -> None:
    with pytest.raises(RuntimeError, match="1 of 2 trials failed"):
        _run()
    assert refit_tags == []  # the outer experiment never started


def test__run_experiment__require_all_trials_off__refits_the_surviving_config(
    refit_tags: list[str],
) -> None:
    summary = _run(require_all_trials=False)

    assert refit_tags == ["default"]
    assert summary.tuned is not None and summary.tuned.test_score == 0.75


def test__run_experiment__cache_dir__passes_one_resolved_config(
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
    runner.run_experiment(
        _MODEL,
        "rel-f1",
        "driver-dnf",
        search_space=object(),
        download=False,
        evaluate_test=False,
        cache_dir=tmp_path,
    )
    assert seen == [(CacheConfig(tmp_path, "raise"), _IDENTITY.for_phase("inner"))]
