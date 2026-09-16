"""Tests for metric direction and primary-metric selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from relbench.base import TaskType
from relbench.metrics import roc_auc

from relarena.metrics import (
    _METRICS,
    Metric,
    get_metric,
    is_higher_better,
    primary_metric,
    to_metric_error,
)


def test_primary_metric_by_task_type() -> None:
    def task(tt: TaskType) -> SimpleNamespace:
        return SimpleNamespace(task_type=tt)

    assert primary_metric(task(TaskType.BINARY_CLASSIFICATION)).__name__ == "roc_auc"
    assert primary_metric(task(TaskType.REGRESSION)).__name__ == "mae"


def test_primary_metric_directions() -> None:
    assert is_higher_better("roc_auc") is True
    assert is_higher_better("mae") is False


def test_get_metric_resolves_name_callable_and_object() -> None:
    m = get_metric("roc_auc")
    assert isinstance(m, Metric)
    assert m.name == "roc_auc"
    assert m.higher_is_better is True
    assert m.optimum == 1.0
    # a metric callable resolves by its __name__
    assert get_metric(roc_auc) is m
    # a Metric passes through unchanged
    assert get_metric(m) is m


def test_get_metric_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_metric("not_a_metric")


def test_metric_is_better_respects_direction() -> None:
    assert get_metric("roc_auc").is_better(0.9, 0.8) is True
    assert get_metric("mae").is_better(0.1, 0.2) is True


@pytest.mark.parametrize(
    ("score", "metric", "error"),
    [
        # higher-is-better: error = optimum (1) - score
        (0.9, "average_precision", 0.1),
        (0.8, "roc_auc", 0.2),
        (0.7, "accuracy", 0.3),
        (0.6, "f1", 0.4),
        (0.0, "r2", 1.0),
        # lower-is-better: the score is already its own error (optimum 0)
        (0.5, "mae", 0.5),
        (2.0, "rmse", 2.0),
    ],
)
def test_to_metric_error_converts_score_to_error(
    score: float, metric: str, error: float
) -> None:
    result = to_metric_error(score, metric)
    assert result == pytest.approx(error)
    assert result >= 0.0  # bencheval requires a non-negative metric_error


def test_to_metric_error_perfect_score_is_zero() -> None:
    # The optimum (best achievable score) is zero error for every known metric.
    for metric in _METRICS.values():
        assert metric.to_error(metric.optimum) == pytest.approx(0.0)


def test_to_metric_error_propagates_nan() -> None:
    # NaN scores (e.g. roc_auc on a degenerate window) must stay NaN, not raise.
    assert np.isnan(to_metric_error(float("nan"), "roc_auc"))


def test_to_metric_error_unknown_metric_raises() -> None:
    with pytest.raises(KeyError):
        to_metric_error(0.5, "not_a_metric")
