"""Tests for the SQL-defined entity task (label query execution)."""

from __future__ import annotations

import pandas as pd
import pytest
from relbench.base import Database, Table

from relarena.userdb.task import UserEntityTask


def _drivers_db() -> Database:
    """A minimal one-table RelBench Database to register under the query."""
    drivers = Table(
        df=pd.DataFrame({"driver_id": [0, 1, 2]}),
        fkey_col_to_pkey_table={},
        pkey_col="driver_id",
        time_col=None,
    )
    return Database({"drivers": drivers})


def _task(query: str) -> UserEntityTask:
    """A UserEntityTask carrying `query`, bypassing the downloading __init__."""
    task = object.__new__(UserEntityTask)
    task.entity_table = "drivers"
    task.entity_col = "driver_id"
    task.time_col = "t"
    task.target_col = "label"
    task.timedelta = pd.Timedelta("30 days")
    task._query = query
    return task


def test__make_table__literal_braces__preserved_and_timedelta_filled() -> None:
    """Literal braces in the SQL survive; only `{timedelta}` is substituted.

    A JSON string literal ('{"type":"x"}') carries braces that str.format would
    misparse as replacement fields - the reason the substitution is a literal
    .replace. The INTERVAL '{timedelta}' proves the one real placeholder is filled.
    """
    query = """
        SELECT
            ts.timestamp AS t,
            d.driver_id AS driver_id,
            CASE
                WHEN '{"type":"x"}' LIKE '%type%'
                     AND ts.timestamp <= ts.timestamp + INTERVAL '{timedelta}'
                THEN 1 ELSE 0
            END AS label
        FROM timestamp_df ts
        CROSS JOIN drivers d
    """
    timestamps = pd.to_datetime(["2020-01-01", "2020-02-01"])

    table = _task(query).make_table(_drivers_db(), timestamps)

    df = table.df
    assert set(df.columns) == {"t", "driver_id", "label"}
    assert len(df) == len(timestamps) * 3  # one row per (anchor, driver)
    assert (df["label"] == 1).all()
    assert table.fkey_col_to_pkey_table == {"driver_id": "drivers"}


def test__make_table__output__sorted_by_time_then_entity() -> None:
    """Rows come back sorted by (time, entity), regardless of the query's order."""
    query = """
        SELECT ts.timestamp AS t, d.driver_id AS driver_id, 0 AS label
        FROM timestamp_df ts CROSS JOIN drivers d
        ORDER BY d.driver_id DESC
    """
    timestamps = pd.to_datetime(["2020-02-01", "2020-01-01"])

    df = _task(query).make_table(_drivers_db(), timestamps).df

    expected = df.sort_values(["t", "driver_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df, expected)


def test__make_table__duplicate_time_entity_keys__raises() -> None:
    """A query emitting the same (time, entity) twice is rejected.

    The (time, entity) label key must be unique.
    """
    query = """
        SELECT ts.timestamp AS t, d.driver_id AS driver_id, 0 AS label
        FROM timestamp_df ts CROSS JOIN drivers d
        UNION ALL
        SELECT ts.timestamp AS t, d.driver_id AS driver_id, 0 AS label
        FROM timestamp_df ts CROSS JOIN drivers d
    """
    timestamps = pd.to_datetime(["2020-01-01"])

    with pytest.raises(ValueError, match="duplicate"):
        _task(query).make_table(_drivers_db(), timestamps)


def test__make_table__extra_output_column__raises() -> None:
    """A query returning a column beyond the three-column contract is rejected.

    Extra columns would silently leak into the entity featurizer as model inputs,
    so make_table must fail loud rather than pass them through.
    """
    query = """
        SELECT
            ts.timestamp AS t,
            d.driver_id AS driver_id,
            0 AS label,
            42 AS leaked
        FROM timestamp_df ts
        CROSS JOIN drivers d
    """
    timestamps = pd.to_datetime(["2020-01-01"])

    with pytest.raises(ValueError, match="exactly.*time_col, entity_col, target_col"):
        _task(query).make_table(_drivers_db(), timestamps)


def test__make_table__source_table_named_timestamp_df__raises() -> None:
    """A source table named timestamp_df collides with the reserved anchor relation."""
    clash = Table(
        df=pd.DataFrame({"timestamp_df_id": [0]}),
        fkey_col_to_pkey_table={},
        pkey_col="timestamp_df_id",
        time_col=None,
    )
    drivers = _drivers_db().table_dict["drivers"]
    db = Database({"drivers": drivers, "timestamp_df": clash})
    query = (
        "SELECT ts.timestamp AS t, d.driver_id AS driver_id, 0 AS label "
        "FROM timestamp_df ts CROSS JOIN drivers d"
    )

    with pytest.raises(ValueError, match="reserved table name"):
        _task(query).make_table(db, pd.to_datetime(["2020-01-01"]))
