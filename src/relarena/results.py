"""Result schema.

A `TrialResult` is the atomic record: one config fit & evaluated on one
`(task, seed)`. It stores configurations, metrics, **wall-clock times**, and
optional predictions as useful metadata for downstream analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from relarena.runner import ExperimentSummary


def config_id_for(config: dict[str, Any]) -> str:
    """A short, deterministic id for a hyperparameter config (order-independent)."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


@dataclass
class TrialResult:
    """Outcome of fitting & evaluating ONE config (within an ExperimentSummary).

    Identity — model / dataset / task / seed / metric — is owned by the enclosing
    `ExperimentSummary` (every trial in a summary shares
    it), so it lives there once instead of being duplicated on each trial; a
    `TrialResult` records only the per-config result.
    """

    config: dict[str, Any]
    config_id: str
    config_tag: str  # "default" or "r{i}" — identifies the default vs random configs

    # The primary (selection) metric's value. `val_score` comes from the
    # train-only model (the selection signal); `test_score` is filled only for
    # the selected config, from the model's final-fit regime (see the runner).
    val_score: float | None = None
    test_score: float | None = None

    # All of the task's native metrics (plus the primary), keyed by metric name.
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Wall-clock seconds, split by phase: tuning (train-only fit + val predict,
    # per config) and final fit (fit + test predict for selected/default configs).
    # The refit-named fields are None for configs that did not receive a final fit.
    fit_time_tuning: float = 0.0
    predict_time_tuning: float = 0.0
    fit_time_refit: float | None = None
    predict_time_refit: float | None = None

    # Optional prediction metadata (not serialized to the summary DataFrame).
    # Shapes follow EntityTask.evaluate's contract.
    val_pred: np.ndarray | None = field(default=None, repr=False)
    test_pred: np.ndarray | None = field(default=None, repr=False)

    # Populated with a traceback string if the trial failed; `None` on success.
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the trial succeeded (no error)."""
        return self.error is None


#: Prediction-array fields dropped when flattening trials to a tidy DataFrame.
#: The native-metric dicts (`val_metrics` / `test_metrics`) are *not* dropped
#: — they're expanded into `val_<metric>` / `test_<metric>` columns.
_ARRAY_FIELDS = {"val_pred", "test_pred"}
_METRIC_DICT_FIELDS = {"val_metrics", "test_metrics"}


def trials_to_dataframe(trials: list[TrialResult]) -> pd.DataFrame:
    """Flatten trials into a tidy DataFrame.

    Cached prediction arrays are dropped. The native-metric dicts are expanded
    into `val_<metric>` / `test_<metric>` columns, so every metric the task
    evaluated is recorded — not just the primary `val_score` / `test_score`.
    Tasks with different native metrics yield sparse columns (NaN where a metric
    does not apply), which is fine for cross-task aggregation.
    """
    import pandas as pd

    skip = _ARRAY_FIELDS | _METRIC_DICT_FIELDS
    rows = []
    for t in trials:
        row = {f.name: getattr(t, f.name) for f in fields(t) if f.name not in skip}
        for prefix in ("val", "test"):
            for name, value in getattr(t, f"{prefix}_metrics").items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


# Run-identity columns prepended to every per-config row.
_IDENTITY_COLS = (
    "id",
    "model",
    "dataset",
    "task",
    "task_type",
    "seed",
    "n_trials",
    "metric",
)


def summary_to_dataframe(
    summary: ExperimentSummary,
    *,
    job_id: str | None = None,
) -> pd.DataFrame:
    """One row per evaluated config (the full search), with identity columns.

    The single results schema shared by the `relarena` CLI and batch sweeps:
    every trial keeps its native scores and all metrics (`val_<m>` /
    `test_<m>`, from `trials_to_dataframe`) plus phase times; only the
    val-selected config (`selected=True`) and the zero-tuning default carry a
    refit `test_score`. The hyperparameter `config` is JSON-encoded for
    portability. The run's identity (model/dataset/task/task_type/seed/n_trials/
    metric) comes straight off the self-describing `summary`; `job_id` is a
    batch run's cache key, omitted for in-process CLI runs (so the `id` column
    is then absent).
    """
    df = trials_to_dataframe(summary.trials)
    if "config" in df.columns:
        df["config"] = df["config"].map(
            lambda c: json.dumps(c, sort_keys=True) if isinstance(c, dict) else c
        )
    tuned_id = summary.tuned.config_id if summary.tuned is not None else None
    df["selected"] = df.get("config_id") == tuned_id
    if job_id is not None:
        df["id"] = job_id
    df["model"] = summary.model_name
    df["dataset"] = summary.dataset
    df["task"] = summary.task_name
    df["task_type"] = summary.task_type.name
    df["seed"] = summary.seed
    df["n_trials"] = summary.n_trials
    df["metric"] = summary.metric_name
    front = [c for c in _IDENTITY_COLS if c in df.columns] + ["selected"]
    return df[front + [c for c in df.columns if c not in front]]
