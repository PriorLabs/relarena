"""Metric metadata and primary-metric selection.

RelBench exposes metrics as plain callables (`relbench.metrics`) and attaches a
list of them to each task. RelArena adds two things on top:

  * a `Metric` registry carrying each metric's **direction** (is higher
    better?) and **optimum** (its best achievable value, in the metric's native
    scale), used to rank trials — and, downstream, to convert scores to errors;
  * a **primary metric per task type** — the one we tune and select models on.
    We deliberately do *not* use `task.metrics[0]` for this: that ordering is
    inconsistent across tasks (binary tasks lead with either `average_precision`
    or `accuracy`) and doesn't match RelBench's headline metrics. The primary
    metric need not be one of the task's native metrics — the tuner scores it by
    passing it to `task.evaluate(pred, metrics=[...])`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from relbench.base import EntityTask, TaskType
from relbench.metrics import mae, roc_auc


@dataclass(frozen=True)
class Metric:
    """A metric's metadata: its name, direction, and best achievable value.

    Bundles what used to live in parallel name-keyed dicts. `optimum` is in the
    metric's **native scale** (mirrors AutoGluon's `Scorer.optimum`): bounded
    higher-is-better scores top out at `1.0`; error-like lower-is-better metrics
    bottom out at `0.0`.
    """

    name: str
    higher_is_better: bool
    optimum: float

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """Whether `candidate` beats `incumbent` under this metric's direction."""
        if self.higher_is_better:
            return candidate > incumbent
        return candidate < incumbent

    def to_error(self, score: float) -> float:
        """Convert a `score` (in this metric's native direction) to an *error*.

        Mirrors TabArena/AutoGluon's `Scorer.convert_score_to_error` so RelArena
        results feed straight into `bencheval`, which expects a lower-is-better
        `metric_error` with `error >= 0`:

            error = sign * (optimum - score)

        `sign` is `+1` for higher-is-better metrics and `-1` otherwise, so a
        higher-is-better score becomes `optimum - score` (e.g. `roc_auc` 0.9
        -> 0.1) and a lower-is-better metric is already its own error (`mae` 0.5
        -> 0.5). A perfect score maps to `0.0`; `NaN` propagates (callers must
        drop/impute those rows before bencheval's no-null check).
        """
        sign = 1.0 if self.higher_is_better else -1.0
        return sign * (self.optimum - float(score))


#: Every metric RelArena knows: its primary metrics plus the native metrics of
#: in-scope entity tasks (recorded for reporting). `get_metric` raises on names
#: that are not registered here.
_METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        # binary metrics: [average_precision, accuracy, f1, roc_auc] (roc_auc primary)
        Metric("roc_auc", higher_is_better=True, optimum=1.0),
        Metric("average_precision", higher_is_better=True, optimum=1.0),
        Metric("accuracy", higher_is_better=True, optimum=1.0),
        Metric("f1", higher_is_better=True, optimum=1.0),
        # regression metrics: [r2, mae, rmse] (mae primary)
        Metric("r2", higher_is_better=True, optimum=1.0),
        Metric("mae", higher_is_better=False, optimum=0.0),
        Metric("rmse", higher_is_better=False, optimum=0.0),
    )
}

#: The metric each task type is tuned and selected on.
PRIMARY_METRIC_BY_TASK_TYPE: dict[TaskType, Callable[..., float]] = {
    TaskType.BINARY_CLASSIFICATION: roc_auc,
    TaskType.REGRESSION: mae,
}

MetricLike = Union[Callable[..., float], str, Metric]


def _metric_name(metric: MetricLike) -> str:
    if isinstance(metric, Metric):
        return metric.name
    return metric if isinstance(metric, str) else metric.__name__


def get_metric(metric: MetricLike) -> Metric:
    """Look up the `Metric` for a name, metric callable, or `Metric`.

    Raises `KeyError` for unknown metrics — add them to `_METRICS` rather than
    silently assuming a direction/optimum.
    """
    if isinstance(metric, Metric):
        return metric
    name = _metric_name(metric)
    if name not in _METRICS:
        raise KeyError(
            f"Unknown metric '{name}'; add it to _METRICS in relarena/metrics.py"
        )
    return _METRICS[name]


def is_higher_better(metric: MetricLike) -> bool:
    """Return whether a higher value of `metric` is better."""
    return get_metric(metric).higher_is_better


def primary_metric(task: EntityTask) -> Callable[..., float]:
    """The metric used to tune and select models for `task`, by its task type.

    Models are ranked on this metric; the tuner additionally records all of the
    task's native metrics. Raises if the task type has no configured primary.
    """
    task_type = task.task_type
    if task_type not in PRIMARY_METRIC_BY_TASK_TYPE:
        raise ValueError(f"No primary metric configured for task type {task_type}.")
    return PRIMARY_METRIC_BY_TASK_TYPE[task_type]


def is_better(candidate: float, incumbent: float, metric: MetricLike) -> bool:
    """Return whether `candidate` beats `incumbent` under `metric`'s direction."""
    return get_metric(metric).is_better(candidate, incumbent)


def to_metric_error(score: float, metric: MetricLike) -> float:
    """Convert `score` (in `metric`'s native direction) to a non-negative error.

    Thin wrapper over `Metric.to_error` for the common DataFrame path — the
    bencheval adapter applies it per row to build the `metric_error` column.
    """
    return get_metric(metric).to_error(score)
