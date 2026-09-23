"""RelBench adapter for Nori-Rel's native relational featurizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import pandas as pd
from relbench.base import Database, EntityTask, Table

from .feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
)

_TARGET_HISTORY_TABLE = "__nori_rel_target_history__"


@dataclass(frozen=True)
class _TaskSignature:
    """Task fields that determine the fitted relational feature plan."""

    entity_table: str
    entity_column: str
    cutoff_column: str | None
    target_column: str
    horizon_ns: int | None

    @classmethod
    def from_task(cls, task: EntityTask) -> _TaskSignature:
        """Normalize a RelBench task to an immutable signature."""
        cutoff = task.time_col
        horizon = getattr(task, "timedelta", None) if cutoff is not None else None
        horizon_ns = None if horizon is None else int(pd.Timedelta(horizon).value)
        return cls(
            entity_table=task.entity_table,
            entity_column=task.entity_col,
            cutoff_column=cutoff,
            target_column=task.target_col,
            horizon_ns=horizon_ns,
        )

    def anchors(self) -> AnchorDefinition:
        """Build the neutral anchor definition consumed by the core planner."""
        horizon = None if self.horizon_ns is None else pd.Timedelta(self.horizon_ns)
        return AnchorDefinition.create(
            entity_table=self.entity_table,
            entity_column=self.entity_column,
            cutoff_column=self.cutoff_column,
            horizon=horizon,
        )


class RelBenchNativeFeaturizer:
    """Fit-freeze native features and materialize aligned RelBench matrices."""

    def __init__(
        self,
        config: NativeFeatureConfig | None = None,
        *,
        with_target_history: bool = True,
    ) -> None:
        """Configure bounded planning and optional strict-past target history."""
        self._config = config
        self._with_target_history = with_target_history
        self._signature: _TaskSignature | None = None
        self._history: pd.DataFrame | None = None
        self._featurizer: NativeRelationalFeaturizer | None = None

    def fit(
        self,
        task: EntityTask,
        database: Database,
        train_table: Table,
        *,
        history_table: Table | None = None,
    ) -> Self:
        """Freeze a plan and a copy of the strictly-past label event stream."""
        signature = _TaskSignature.from_task(task)
        history = self._target_history(
            signature,
            history_table if history_table is not None else train_table,
        )
        view, excluded = _database_view(database, signature, history)
        self._featurizer = NativeRelationalFeaturizer.fit(
            view,
            signature.anchors(),
            self._config,
            excluded_columns=excluded,
        )
        self._signature = signature
        self._history = history
        return self

    def transform(
        self,
        task: EntityTask,
        database: Database,
        table: Table,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return one native feature row per task row in original order."""
        if self._featurizer is None or self._signature is None:
            raise RuntimeError("native RelBench featurizer must be fitted first")
        signature = _TaskSignature.from_task(task)
        if signature != self._signature:
            raise ValueError("RelBench task does not match the fitted feature plan")
        view, _ = _database_view(database, signature, self._history)
        matrix = self._featurizer.transform(
            view,
            _anchor_frame(table, signature),
        )
        return matrix.frame, list(matrix.categorical_columns)

    @property
    def plan_digest(self) -> str:
        """Return the fitted plan digest for cache and provenance records."""
        if self._featurizer is None:
            raise RuntimeError("native RelBench featurizer must be fitted first")
        return self._featurizer.plan.digest

    def _target_history(
        self,
        signature: _TaskSignature,
        table: Table,
    ) -> pd.DataFrame | None:
        if not self._with_target_history or signature.cutoff_column is None:
            return None
        columns = [
            signature.entity_column,
            signature.cutoff_column,
            signature.target_column,
        ]
        if len(set(columns)) != len(columns):
            raise ValueError("entity, cutoff, and target columns must be distinct")
        missing = [column for column in columns if column not in table.df]
        if missing:
            raise ValueError(f"target history columns are missing: {missing}")
        history = table.df.loc[:, columns].reset_index(drop=True).copy()
        keys = [signature.entity_column, signature.cutoff_column]
        if history[keys].isna().any().any():
            raise ValueError("target history keys and cutoffs must be present")
        if history.duplicated(keys).any():
            raise ValueError("target history requires one target per entity and cutoff")
        return history


def _database_view(
    database: Database,
    task: _TaskSignature,
    history: pd.DataFrame | None,
) -> tuple[DatabaseView, dict[str, set[str]]]:
    """Translate public RelBench metadata without retaining RelBench objects."""
    tables = {name: table.df for name, table in database.table_dict.items()}
    if task.entity_table not in tables:
        raise ValueError(f"missing entity table {task.entity_table!r}")
    if history is not None and _TARGET_HISTORY_TABLE in tables:
        raise ValueError(f"reserved table name {_TARGET_HISTORY_TABLE!r} is in use")

    primary_keys = {
        name: table.pkey_col
        for name, table in database.table_dict.items()
        if table.pkey_col is not None
    }
    try:
        entity_key = primary_keys[task.entity_table]
    except KeyError as exc:
        raise ValueError("entity table requires a single-column primary key") from exc

    foreign_keys: list[tuple[str, str, str, str]] = []
    for child_name, table in database.table_dict.items():
        for child_column, parent_name in table.fkey_col_to_pkey_table.items():
            try:
                parent_column = primary_keys[parent_name]
            except KeyError as exc:
                raise ValueError(
                    f"foreign key {child_name}.{child_column} refers to "
                    f"{parent_name!r} without a primary key"
                ) from exc
            foreign_keys.append((child_name, child_column, parent_name, parent_column))

    time_columns = {
        name: table.time_col
        for name, table in database.table_dict.items()
        if table.time_col is not None
    }
    # RelBench labels live in the task table, not in ``database``. Preserve
    # same-named columns on related tables: they are often valid historical
    # facts (for example, prior ``results.position`` values). A timeless direct
    # entity-table collision is the only unsafe database source here.
    root = tables[task.entity_table]
    excluded = (
        {task.entity_table: {task.target_column}}
        if task.target_column in root
        and (task.cutoff_column is None or task.entity_table not in time_columns)
        else {}
    )
    if history is not None:
        if task.cutoff_column is None:
            raise RuntimeError("target history requires a temporal task")
        tables[_TARGET_HISTORY_TABLE] = history
        time_columns[_TARGET_HISTORY_TABLE] = task.cutoff_column
        foreign_keys.append(
            (
                _TARGET_HISTORY_TABLE,
                task.entity_column,
                task.entity_table,
                entity_key,
            )
        )

    return (
        DatabaseView.from_parts(
            tables=tables,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns=time_columns,
        ),
        excluded,
    )


def _anchor_frame(table: Table, task: _TaskSignature) -> pd.DataFrame:
    """Copy only entity and cutoff columns so labels cannot enter execution."""
    columns = [task.entity_column]
    if task.cutoff_column is not None:
        columns.append(task.cutoff_column)
    missing = [column for column in columns if column not in table.df]
    if missing:
        raise ValueError(f"RelBench anchor columns are missing: {missing}")
    return table.df.loc[:, columns].reset_index(drop=True).copy()


__all__ = ["RelBenchNativeFeaturizer"]
