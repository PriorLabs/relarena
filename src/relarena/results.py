"""Model and system result schemas.

Models produce one `TrialResult` per harness-selected configuration. Systems
produce one `SystemResult` for their complete internal procedure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from relarena.runner import ExperimentSummary, SystemExperimentSummary


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


@dataclass
class SystemResult:
    """Outcome of one end-to-end system run.

    Systems do not expose harness-selected configurations or validation scores.
    Their complete internal procedure is represented by a final test result and
    one total wall-clock time.
    """

    test_score: float | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)
    time_total: float = 0.0
    test_pred: np.ndarray | None = field(default=None, repr=False)


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
    "kind",
    "dataset",
    "task",
    "task_type",
    "seed",
    "n_trials",
    "metric",
)


def summary_to_dataframe(
    summary: ExperimentSummary | SystemExperimentSummary,
    *,
    job_id: str | None = None,
) -> pd.DataFrame:
    """Flatten a model or system experiment into the shared results frame.

    Model summaries contain one row per trial, including config, validation
    metrics, and phase timings. A system summary contains one selected row with
    its final test metrics and `time_total`; it has no synthetic config,
    validation score, or trial count. `job_id` is an optional batch cache key.
    """
    import pandas as pd

    # Import at call time to avoid a results ↔ runner import cycle.
    from relarena.runner import SystemExperimentSummary

    if isinstance(summary, SystemExperimentSummary):
        result = summary.result
        row: dict[str, Any] = {
            "model": summary.system_name,
            "kind": "system",
            "dataset": summary.dataset,
            "task": summary.task_name,
            "task_type": summary.task_type.name,
            "seed": summary.seed,
            "metric": summary.metric_name,
            "selected": True,
            "test_score": result.test_score,
            "time_total": result.time_total,
            "peak_rss_gib": summary.peak_rss_gib,
        }
        if job_id is not None:
            row["id"] = job_id
        row.update(
            {f"test_{name}": value for name, value in result.test_metrics.items()}
        )
        front = [c for c in _IDENTITY_COLS if c in row] + ["selected"]
        remaining = [key for key in row if key not in front]
        return pd.DataFrame([{column: row[column] for column in front + remaining}])

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
    df["kind"] = summary.kind
    df["dataset"] = summary.dataset
    df["task"] = summary.task_name
    df["task_type"] = summary.task_type.name
    df["seed"] = summary.seed
    df["n_trials"] = summary.n_trials
    df["metric"] = summary.metric_name
    df["peak_rss_gib"] = summary.peak_rss_gib
    front = [c for c in _IDENTITY_COLS if c in df.columns] + ["selected"]
    return df[front + [c for c in df.columns if c not in front]]
