"""User-facing task specification for the SQL-based predictive interface."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from relbench.base import TaskType

from relarena.userdb.predict import EntitySelector

#: String aliases accepted for `task_type`, restricted to the entity task types
#: RelArena supports.
_TASK_TYPE_BY_NAME: dict[str, TaskType] = {
    "binary_classification": TaskType.BINARY_CLASSIFICATION,
    "regression": TaskType.REGRESSION,
}


@dataclass(frozen=True)
class PredictiveTaskSpec:
    """One predictive entity task: label SQL, evaluation split, prediction target.

    The `query` is a DuckDB statement that joins `timestamp_df` (a single column
    `timestamp` holding the anchor times) against the database tables - referenced by
    their names - and returns exactly `time_col`, `entity_col` and `target_col`,
    computing the label over the forward window `(t, t + timedelta]`. The literal
    `{timedelta}` in the query is replaced with the string form of `timedelta`, e.g.
    `INTERVAL '{timedelta}'`.

    `val_timestamp` / `test_timestamp` are the tuning and final split cutoffs;
    `entities` / `at_timestamp` say what to predict at inference (default: all
    entities at `test_timestamp`). A later `at_timestamp` changes the prediction
    anchor but not the RelBench-style feature snapshot, which remains frozen at
    `test_timestamp`. The database schema lives separately in a `DatabaseSpec`, so
    one database can back many tasks.

    `task_type` accepts the RelBench enum or the strings `"binary_classification"` /
    `"regression"`; `timedelta` accepts a `pandas.Timedelta` or anything it can parse.
    """

    entity_table: str
    entity_col: str
    time_col: str
    target_col: str
    task_type: TaskType
    timedelta: pd.Timedelta
    query: str
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    num_eval_timestamps: int = 1
    at_timestamp: pd.Timestamp | None = None
    entities: EntitySelector = "all"

    def __post_init__(self) -> None:
        """Normalize `task_type`, `timedelta`, and the split/anchor timestamps."""
        if isinstance(self.task_type, str):
            try:
                task_type = _TASK_TYPE_BY_NAME[self.task_type]
            except KeyError:
                raise ValueError(
                    f"Unsupported task_type {self.task_type!r}; expected one of "
                    f"{sorted(_TASK_TYPE_BY_NAME)} or a RelBench TaskType."
                ) from None
            object.__setattr__(self, "task_type", task_type)
        elif self.task_type not in _TASK_TYPE_BY_NAME.values():
            raise ValueError(
                f"Unsupported task_type {self.task_type!r}; RelArena entity tasks "
                f"are binary classification or regression only."
            )

        if not isinstance(self.timedelta, pd.Timedelta):
            object.__setattr__(self, "timedelta", pd.Timedelta(self.timedelta))
        object.__setattr__(self, "val_timestamp", pd.Timestamp(self.val_timestamp))
        object.__setattr__(self, "test_timestamp", pd.Timestamp(self.test_timestamp))
        if self.at_timestamp is not None:
            object.__setattr__(self, "at_timestamp", pd.Timestamp(self.at_timestamp))
