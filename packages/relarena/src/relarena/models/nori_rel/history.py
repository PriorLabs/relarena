"""Strict-past target lags for the Nori-Rel training-profile route.

Adapted from RelArena's Apache-2.0 TabPFN-Rel history feature implementation,
Copyright 2026 PriorLabs GmbH. See ``ATTRIBUTION.md``.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from relbench.base import EntityTask, Table


class StrictPastHistory:
    """Attach the latest target values strictly before each row's cutoff."""

    def __init__(self, n_lags: int) -> None:
        """Initialize an unfitted history transformer."""
        if n_lags < 1:
            raise ValueError("n_lags must be positive")
        self.n_lags = n_lags
        self._history: pd.DataFrame | None = None

    def fit_transform(
        self,
        features: pd.DataFrame,
        task: EntityTask,
        train_table: Table,
    ) -> pd.DataFrame:
        """Freeze the training look-back pool and augment the training rows."""
        return self.fit(task, train_table).transform(features, task, train_table)

    def fit(self, task: EntityTask, train_table: Table) -> Self:
        """Freeze the full training look-back pool without transforming rows."""
        if task.time_col is None:
            raise ValueError("strict-past history requires a temporal task")
        self._history = _prepare_history(
            train_table.df,
            entity_col=task.entity_col,
            time_col=task.time_col,
            target_col=task.target_col,
            n_lags=self.n_lags,
        )
        return self

    def transform(
        self,
        features: pd.DataFrame,
        task: EntityTask,
        table: Table,
    ) -> pd.DataFrame:
        """Augment rows using only the frozen training look-back pool."""
        if self._history is None:
            raise RuntimeError("strict-past history must be fitted before transform")
        if task.time_col is None:
            raise ValueError("strict-past history requires a temporal task")
        return _attach_history(
            features,
            table.df,
            self._history,
            entity_col=task.entity_col,
            time_col=task.time_col,
            n_lags=self.n_lags,
        )


def _prepare_history(
    labels: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    target_col: str,
    n_lags: int,
) -> pd.DataFrame:
    """Precompute lag values and timestamps in a fit-frozen pool."""
    required = [entity_col, time_col, target_col]
    missing = [column for column in required if column not in labels]
    if missing:
        raise ValueError(f"strict-past training columns are missing: {missing}")
    history = labels[[entity_col, time_col, target_col]].copy()
    history[time_col] = pd.to_datetime(history[time_col])
    if history[entity_col].isna().any() or history[time_col].isna().any():
        raise ValueError("strict-past training keys and cutoffs must be present")
    history[entity_col] = history[entity_col].astype(str)
    history = history.sort_values([entity_col, time_col], kind="mergesort").reset_index(
        drop=True
    )
    grouped = history.groupby(entity_col, sort=False)
    for lag in range(1, n_lags + 1):
        history[f"_lag{lag}_value"] = grouped[target_col].shift(lag - 1)
        history[f"_lag{lag}_time"] = grouped[time_col].shift(lag - 1)
    columns = [entity_col, time_col]
    columns.extend(f"_lag{lag}_value" for lag in range(1, n_lags + 1))
    columns.extend(f"_lag{lag}_time" for lag in range(1, n_lags + 1))
    return (
        history[columns].sort_values(time_col, kind="mergesort").reset_index(drop=True)
    )


def _attach_history(
    features: pd.DataFrame,
    rows: pd.DataFrame,
    history: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    n_lags: int,
) -> pd.DataFrame:
    """Join frozen history with strict-less-than cutoff semantics."""
    if len(features) != len(rows):
        raise ValueError(
            f"strict-past history row mismatch: features={len(features)} "
            f"rows={len(rows)}"
        )

    required = [entity_col, time_col]
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"strict-past query columns are missing: {missing}")
    additions = [
        name
        for lag in range(1, n_lags + 1)
        for name in (f"target_lag{lag}", f"target_lag{lag}_age_days")
    ]
    collisions = [column for column in additions if column in features]
    if collisions:
        raise ValueError(f"strict-past output columns already exist: {collisions}")

    position = "__nori_rel_row__"
    anchors = rows[[entity_col, time_col]].reset_index(drop=True).copy()
    anchors[position] = np.arange(len(anchors), dtype=np.int64)
    anchors[time_col] = pd.to_datetime(anchors[time_col])
    if anchors[entity_col].isna().any() or anchors[time_col].isna().any():
        raise ValueError("strict-past query keys and cutoffs must be present")
    anchors[entity_col] = anchors[entity_col].astype(str)
    anchors = anchors.sort_values(time_col, kind="mergesort")
    merged = pd.merge_asof(
        anchors,
        history,
        on=time_col,
        by=entity_col,
        allow_exact_matches=False,
        direction="backward",
    )

    row_positions = merged[position].to_numpy(dtype=np.int64)
    output = features.reset_index(drop=True).copy()
    for lag in range(1, n_lags + 1):
        values = np.empty(len(merged), dtype=float)
        values[row_positions] = pd.to_numeric(
            merged[f"_lag{lag}_value"], errors="coerce"
        ).to_numpy(dtype=float)
        ages = np.empty(len(merged), dtype=float)
        ages[row_positions] = (
            (merged[time_col] - merged[f"_lag{lag}_time"]) / pd.Timedelta("1D")
        ).to_numpy(dtype=float)
        output[f"target_lag{lag}"] = values
        output[f"target_lag{lag}_age_days"] = ages
    return output


__all__ = ["StrictPastHistory"]
