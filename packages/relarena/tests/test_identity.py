"""Tests for optional preprocessing run identities."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from relbench.base import Database, Table

from relarena.identity import (
    RunIdentity,
    database_schema_fingerprint,
    relbench_run_identity,
    task_spec_fingerprint,
)


def test__relbench_run_identity__recorded_task__is_complete_and_stable() -> None:
    first = relbench_run_identity("rel-f1", "driver-dnf")
    second = relbench_run_identity("rel-f1", "driver-dnf")
    assert first == second
    assert first.dataset_fingerprint is not None
    assert len(first.dataset_fingerprint.split("-")) == 2
    assert first.task_fingerprint is not None
    assert len(first.task_fingerprint) == 16


def test__run_identity__phase_copy__does_not_mutate_base() -> None:
    base = RunIdentity("dataset", "db", "task", "labels")
    assert base.for_phase("inner").phase == "inner"
    assert base.phase is None


def _database(dtype: str = "int64") -> Database:
    return Database(
        {
            "entities": Table(
                df=pd.DataFrame({"id": pd.Series([1, 2], dtype=dtype)}),
                fkey_col_to_pkey_table={},
                pkey_col="id",
                time_col=None,
            )
        }
    )


def test__database_schema_fingerprint__ignores_rows_but_guards_dtype() -> None:
    same_schema = Database(
        {
            "entities": Table(
                df=pd.DataFrame({"id": pd.Series([9], dtype="int64")}),
                fkey_col_to_pkey_table={},
                pkey_col="id",
                time_col=None,
            )
        }
    )
    assert database_schema_fingerprint(_database()) == database_schema_fingerprint(
        same_schema
    )
    assert database_schema_fingerprint(_database()) != database_schema_fingerprint(
        _database("float64")
    )


def _task_spec(**overrides: object) -> SimpleNamespace:
    fields = {
        "entity_table": "drivers",
        "entity_col": "driver_id",
        "time_col": "date",
        "target_col": "dnf",
        "task_type": "binary_classification",
        "timedelta": "30 days",
        "query": "SELECT * FROM labels",
        "val_timestamp": "2005-01-01",
        "test_timestamp": "2005-02-01",
        "num_eval_timestamps": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test__task_spec_fingerprint__is_stable_and_guards_training_semantics() -> None:
    assert task_spec_fingerprint(_task_spec()) == task_spec_fingerprint(_task_spec())
    assert task_spec_fingerprint(_task_spec()) != task_spec_fingerprint(
        _task_spec(timedelta="60 days")
    )
