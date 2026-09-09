"""Tests for optional preprocessing run identities."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from relbench.base import Database, Table

from relarena_core.identity import (
    RunIdentity,
    database_schema_fingerprint,
    metadata_fingerprint,
    task_spec_fingerprint,
)


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


def test_metadata_fingerprint_is_order_independent_and_sensitive_to_values() -> None:
    first = metadata_fingerprint({"b": [2, 3], "a": 1})
    assert first == metadata_fingerprint({"a": 1, "b": [2, 3]})
    assert first != metadata_fingerprint({"a": 1, "b": [3, 2]})
    assert len(first) == 16
    int(first, 16)
