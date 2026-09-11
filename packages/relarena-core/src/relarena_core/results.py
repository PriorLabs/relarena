"""Model and system result schemas.

Models produce one `TrialResult` per harness-selected configuration. Systems
produce one `SystemResult` for their complete internal procedure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
