"""Data sources and splits.

RelArena's fairness protocol is *nested temporal validation* (see
`docs/temporal-validation.md`):

  * an **inner** split — fit on `train`, score on `val`, with the database
    censored at `val_timestamp` — used to tune and select a config;
  * an **outer** split — perform the model's final-fit regime, score on `test`,
    and use a database censored at `test_timestamp` — used for the final number.

The crucial invariant is that *the correct DB censoring is part of the split's
definition*: an inner split must carry a val-censored DB, an outer split a
test-censored one. This module makes that correct-by-construction by bundling
the censored `db` with its label tables in a `Split`, and centralizing
the censoring in `RelBenchDatasetTask` rather than leaving `db.upto(...)` calls
scattered across the runner and tuner.

`RelBenchDatasetTask` supplies named benchmark loading and canonical column
cleaning. Core's `TaskSource` implements the shared supplied-object path and
split construction used by this adapter and standalone predictive queries.
"""

from __future__ import annotations

from typing import Final

from relbench.base import Database, EntityTask, Table
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from relarena.identity import relbench_run_identity
from relarena_core.dataset import (
    TaskSource,
    clean_database,
)

#: rel-ratebeer per-user aggregates computed over the *entire* rating history, so a
#: row can encode information from after its own timestamp.
TIME_LEAKING_COLUMNS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "rel-ratebeer": {
        "users": (
            "max_beer_rating",
            "min_beer_rating",
            "max_place_rating",
            "min_place_rating",
            "place_first_rating",
            "place_last_rating",
        ),
    },
}


def drop_noncanonical_columns(db: Database, dataset_name: str) -> Database:
    """Return `db` reduced to its canonical benchmarking feature set.

    Drops, per the policy in https://github.com/snap-stanford/relbench/issues/373,
    fully-NaN columns and `Unnamed: N` row-index artifacts from every table, plus
    rel-ratebeer's time-leaking user aggregates (`TIME_LEAKING_COLUMNS`).
    Sparse-but-populated columns are kept: models are expected to be robust to them.

    Affected tables are rebuilt, not mutated: relbench `lru_cache`-s `get_db()`
    and `relarena.checksums` must keep seeing the raw upstream data.
    """
    return clean_database(db, TIME_LEAKING_COLUMNS.get(dataset_name, {}))


def drop_noncanonical_task_columns(
    task: EntityTask, table: Table, dataset_name: str
) -> Table:
    """Reduce a task label table to the task's own columns.

    RelBench's rel-event tasks call `reset_index()` without `drop=True` in
    `make_table`, leaking a stray positional `index` column into their label
    tables. Keep only `time_col`, `entity_col` and - when present, since the
    masked test table omits it - `target_col`, so the artifact never reaches a
    model, a checksum, or a reproduction check.

    rel-event is the only dataset known to carry such a column. Dropping one from
    any other dataset is unexpected - a new upstream artifact, or a real column we
    would be discarding - so raise instead: fix the cause, don't silently drop.
    """
    canonical = {task.time_col, task.entity_col, task.target_col}
    extra = [c for c in table.df.columns if c not in canonical]
    if not extra:
        return table
    if dataset_name != "rel-event":
        raise ValueError(
            f"Task table for {dataset_name!r} has unexpected non-canonical "
            f"column(s) {extra}; only rel-event's stray `index` is known - "
            "investigate rather than dropping."
        )
    keep = [c for c in table.df.columns if c in canonical]
    return Table(
        df=table.df[keep],
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


class RelBenchDatasetTask(TaskSource):
    """Load canonical benchmark data and construct its temporal splits."""

    def __init__(
        self, dataset_name: str, task_name: str, *, download: bool = True
    ) -> None:
        """Load the `(dataset, task)`; downloads first unless `download` is off."""
        self.dataset_name = dataset_name
        self.task_name = task_name
        self._identity = relbench_run_identity(dataset_name, task_name)
        self._dataset = get_dataset(dataset_name, download=download)
        self._task: EntityTask = get_task(dataset_name, task_name, download=download)
        # get_db() is already censored at test_timestamp; inner_split re-censors a
        # copy at val_timestamp via Database.upto.
        self._db = drop_noncanonical_columns(self._dataset.get_db(), dataset_name)
        # get_table masks input columns for the test split by default, dropping the
        # target column so the test labels are never handed out (leakage prevention);
        # train/val come back unmasked. The hidden test labels are recovered only
        # inside EntityTask.evaluate(..., target_table=None) at scoring time.
        self._tables = {
            split: drop_noncanonical_task_columns(
                self._task, self._task.get_table(split), self.dataset_name
            )
            for split in ("train", "val", "test")
        }

    _prepare_db = staticmethod(drop_noncanonical_columns)
    _prepare_table = staticmethod(drop_noncanonical_task_columns)


__all__ = [
    "RelBenchDatasetTask",
    "drop_noncanonical_columns",
    "drop_noncanonical_task_columns",
    "TIME_LEAKING_COLUMNS",
]
