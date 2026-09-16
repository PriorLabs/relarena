"""Constant (optimal-constant) baselines.

Two parameter-free reference points, both predicting the best *constant* for the
task's primary metric:

  * `constant-global` — one global constant for every row (scikit-learn's
    `DummyRegressor` / `DummyClassifier`, wrapped directly so there's no
    hand-rolled constant math):

      - regression  -> median for MAE, mean otherwise
      - binary      -> `strategy="prior"`; the P(y=1) base rate

  * `constant-per-entity` — the same optimal constant computed *per entity* from
    that entity's own training rows, falling back to the global constant only for
    entities with no training history. This isolates how much per-entity
    differentiation buys over a single global guess.

Hard-label metrics (accuracy / f1) recover the mode by thresholding the predicted
probability, so "predict the prior" covers both the single-value and distribution
cases. Predictions are probabilities; the `log_loss` metric (which expects
logits) is not special-cased.

Parameter-free: their search spaces are empty, so each runs only as the default
config.
"""

from __future__ import annotations

import numpy as np
from relbench.base import Database, EntityTask, Table, TaskType
from sklearn.dummy import DummyClassifier, DummyRegressor

from relarena.metrics import primary_metric
from relarena.model import RelArenaModel
from relarena.models._shared.predict_contract import predict_to_contract
from relarena.registry import register_model
from relarena.search_space import SearchSpace

#: Metrics minimized by the median; everything else (MSE/RMSE/R²) by the mean.
_MEDIAN_METRICS = {"mae"}


def _regression_uses_median(task: EntityTask) -> bool:
    """Whether the task's primary regression metric is minimized by the median."""
    return primary_metric(task).__name__ in _MEDIAN_METRICS


# untunable: no space/grid -> only the default config (no overrides) runs
@register_model(search_space=SearchSpace(default_overrides={}))
class DummyBaseline(RelArenaModel):
    """Predicts the optimal global constant for the task's primary metric."""

    name = "constant-global"

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Fit the optimal constant (median/mean or class prior) on `train_table`."""
        y = train_table.df[task.target_col]
        self._task_type = task.task_type
        X = np.zeros((len(y), 1))  # Dummy estimators use X only for the row count

        if task.task_type == TaskType.REGRESSION:
            strategy = "median" if _regression_uses_median(task) else "mean"
            self._est = DummyRegressor(strategy=strategy).fit(
                X, y.to_numpy(dtype=float)
            )
        elif task.task_type == TaskType.BINARY_CLASSIFICATION:
            self._est = DummyClassifier(strategy="prior").fit(X, y.to_numpy())
        else:
            raise ValueError(
                f"DummyBaseline does not support task type {task.task_type}."
            )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Return the fitted constant broadcast to every row of `table`."""
        n = len(table.df)
        X = np.zeros((n, 1))  # Dummy estimators use X only for the row count
        return predict_to_contract(self._est, X, self._task_type)


# untunable: no space/grid -> only the default config (no overrides) runs
@register_model(search_space=SearchSpace(default_overrides={}))
class DummyPerEntityBaseline(RelArenaModel):
    """Predicts each entity's own optimal constant, else the global constant."""

    name = "constant-per-entity"

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Fit a per-entity constant (median/mean or positive rate) on `train_table`.

        A single global constant, fit the same way over all rows, is retained as the
        fallback for entities with no training history.
        """
        df = train_table.df
        self._task_type = task.task_type
        self._entity_col = task.entity_col
        y = df[task.target_col]

        if task.task_type == TaskType.REGRESSION:
            y = y.astype(float)
            grouped = y.groupby(df[task.entity_col])
            if _regression_uses_median(task):
                self._per_entity = grouped.median()
                self._global = float(y.median())
            else:
                self._per_entity = grouped.mean()
                self._global = float(y.mean())
        elif task.task_type == TaskType.BINARY_CLASSIFICATION:
            # y in {0, 1}: the per-entity mean is that entity's positive-class rate.
            y = y.astype(float)
            self._per_entity = y.groupby(df[task.entity_col]).mean()
            self._global = float(y.mean())
        else:
            raise ValueError(
                f"DummyPerEntityBaseline does not support task type {task.task_type}."
            )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Look up each row's per-entity constant; use the global one where absent."""
        preds = table.df[self._entity_col].map(self._per_entity).to_numpy(dtype=float)
        return np.where(np.isnan(preds), self._global, preds)
