"""Tests for the grouped dfs2sql engine (no data download)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlglot import parse_one

pytest.importorskip("fastdfs")

from fastdfs import DFSConfig, compute_dfs_features  # noqa: E402
from fastdfs.utils.type_utils import safe_convert_to_string  # noqa: E402

import relarena.featurization.dfs as dfs_mod  # noqa: E402
import relarena.featurization.dfs2sql_grouped as grouped_mod  # noqa: E402
from relarena.featurization.dfs2sql_grouped import (  # noqa: E402
    GROUPED_DFS2SQL_ENGINE,
    group_temp_table_statements,
)
from tests.featurization.test_dfs import _label_table, _toy_db, _toy_task  # noqa: E402


def _stock_chunk(create_names: list[str], select: str) -> list:
    """One feature's worth of stock output: creates, the select, the drops."""
    creates = [parse_one(f"CREATE TABLE {n} AS SELECT 1") for n in create_names]
    drops = [parse_one(f"DROP TABLE {n}") for n in create_names]
    return [*creates, parse_one(select), *drops]


def test__group_temp_table_statements__shared_creates__built_once() -> None:
    a1, a2, b = "SELECT 1 AS a1", "SELECT 2 AS a2", "SELECT 3 AS b"
    statements = [
        *_stock_chunk(["t0", "t1"], a1),
        *_stock_chunk(["t0", "t2"], b),
        *_stock_chunk(["t0", "t1"], a2),
    ]

    grouped = group_temp_table_statements(statements)

    assert [s.sql() for s in grouped] == [
        "CREATE TABLE t0 AS SELECT 1",
        "CREATE TABLE t1 AS SELECT 1",
        a1,
        a2,
        "DROP TABLE t0",
        "DROP TABLE t1",
        "CREATE TABLE t0 AS SELECT 1",
        "CREATE TABLE t2 AS SELECT 1",
        b,
        "DROP TABLE t0",
        "DROP TABLE t2",
    ]
    # The select objects are passed through, so results map back by identity.
    selects = [s for s in statements if s.sql().startswith("SELECT")]
    assert [s for s in grouped if s.sql().startswith("SELECT")] == [
        selects[0],
        selects[2],
        selects[1],
    ]


def test__group_temp_table_statements__no_creates__unchanged() -> None:
    statements = [parse_one("SELECT 1"), parse_one("SELECT 2")]

    assert group_temp_table_statements(statements) == statements


def test__group_temp_table_statements__unexpected_statement__raises() -> None:
    with pytest.raises(TypeError, match="Insert"):
        group_temp_table_statements([parse_one("INSERT INTO t VALUES (1)")])


def _dfs_input() -> pd.DataFrame:
    df = _label_table([1, 2, 3], ts=["2021-06-01", "2020-02-15", "2021-06-01"]).df
    df = df.drop(columns=["y"])
    df["uid"] = safe_convert_to_string(df["uid"])
    return df


def _compute(engine: str, engine_path: Path) -> pd.DataFrame:
    # A fresh RDB per call: `prepare_features` mutates the one it runs on.
    rdb = dfs_mod._transform_rdb(dfs_mod._build_rdb(_toy_db()))
    task = _toy_task()
    return compute_dfs_features(
        rdb,
        _dfs_input(),
        key_mappings={task.entity_col: f"{task.entity_table}.uid"},
        cutoff_time_column=task.time_col,
        config=DFSConfig(max_depth=2, engine=engine, engine_path=str(engine_path)),
    )


def test__grouped_engine__matches_stock_engine_and_issues_fewer_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: dict[str, int] = {}
    regroup = group_temp_table_statements

    def spy(statements: list) -> list:
        grouped = regroup(statements)
        counts["stock"], counts["grouped"] = len(statements), len(grouped)
        return grouped

    monkeypatch.setattr(grouped_mod, "group_temp_table_statements", spy)

    stock = _compute("dfs2sql", tmp_path / "stock.db")
    grouped = _compute(GROUPED_DFS2SQL_ENGINE, tmp_path / "grouped.db")

    pd.testing.assert_frame_equal(grouped, stock)
    assert counts["grouped"] < counts["stock"]
