"""Tests for building a RelBench database/dataset from a user manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from relarena.userdb.ingest import DatabaseSpec, TableSource, build_dataset


def test__build_dataset__pkey_maps__capture_original_to_reindexed_ids(
    tmp_path: Path,
) -> None:
    """The captured maps mirror reindex: static row order, temporal time order."""
    # Static entity table, string pkeys -> reindexed in row order.
    pd.DataFrame({"seller_id": ["s_a", "s_b", "s_c"]}).to_parquet(
        tmp_path / "sellers.parquet"
    )
    # Temporal table, rows NOT in time order -> reindexed sorted by time_col.
    pd.DataFrame(
        {
            "order_id": ["o_mar", "o_jan", "o_feb"],
            "t": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
        }
    ).to_parquet(tmp_path / "orders.parquet")

    spec = DatabaseSpec(
        tables={
            "sellers": TableSource(
                path=str(tmp_path / "sellers.parquet"), pkey="seller_id"
            ),
            "orders": TableSource(
                path=str(tmp_path / "orders.parquet"), pkey="order_id", time_col="t"
            ),
        }
    )
    ds = build_dataset(spec, val_timestamp="2020-02-15", test_timestamp="2020-03-15")

    sellers = ds.pkey_maps["sellers"]
    assert list(sellers.index) == ["s_a", "s_b", "s_c"]
    assert list(sellers.to_numpy()) == [0, 1, 2]

    orders = ds.pkey_maps["orders"]
    assert list(orders.index) == ["o_jan", "o_feb", "o_mar"]  # time-sorted
    assert list(orders.to_numpy()) == [0, 1, 2]


def test__build_dataset__csv_time_col__parsed_to_datetime(tmp_path: Path) -> None:
    """A declared time_col read from CSV is parsed to datetime, not left as strings."""
    (tmp_path / "sellers.csv").write_text("seller_id\ns_a\ns_b\n")
    (tmp_path / "orders.csv").write_text(
        "order_id,seller_id,t\n0,s_a,2020-01-01\n1,s_b,2020-02-01\n"
    )
    spec = DatabaseSpec(
        tables={
            "sellers": TableSource(
                path=str(tmp_path / "sellers.csv"), pkey="seller_id"
            ),
            "orders": TableSource(
                path=str(tmp_path / "orders.csv"),
                pkey="order_id",
                time_col="t",
                fkeys={"seller_id": "sellers"},
            ),
        }
    )
    ds = build_dataset(spec, val_timestamp="2020-01-15", test_timestamp="2020-02-15")

    assert pd.api.types.is_datetime64_any_dtype(ds._db.table_dict["orders"].df["t"])


def test__build_dataset__columns__subsets_table(tmp_path: Path) -> None:
    """`columns` restricts a table to the listed columns; the rest are dropped."""
    pd.DataFrame(
        {"seller_id": ["s_a", "s_b"], "city": ["x", "y"], "secret": [1, 2]}
    ).to_parquet(tmp_path / "sellers.parquet")
    spec = DatabaseSpec(
        tables={
            "sellers": TableSource(
                path=str(tmp_path / "sellers.parquet"),
                pkey="seller_id",
                columns=["seller_id", "city"],
            )
        }
    )
    ds = build_dataset(spec, val_timestamp="2020-01-01", test_timestamp="2020-02-01")
    assert set(ds._db.table_dict["sellers"].df.columns) == {"seller_id", "city"}


def test__build_dataset__columns_drop_pkey__raises(tmp_path: Path) -> None:
    """`columns` that omits the pkey/time_col/fkey is rejected."""
    pd.DataFrame({"seller_id": ["s_a"], "city": ["x"]}).to_parquet(
        tmp_path / "sellers.parquet"
    )
    spec = DatabaseSpec(
        tables={
            "sellers": TableSource(
                path=str(tmp_path / "sellers.parquet"),
                pkey="seller_id",
                columns=["city"],  # drops the pkey
            )
        }
    )
    with pytest.raises(ValueError, match="must be listed in `columns`"):
        build_dataset(spec, val_timestamp="2020-01-01", test_timestamp="2020-02-01")


def test__database_spec__from_yaml__parses_tables_columns_and_paths(
    tmp_path: Path,
) -> None:
    """DatabaseSpec.from_yaml parses each table's schema, columns and resolved path."""
    (tmp_path / "db.yaml").write_text(
        "drivers:\n"
        "  pkey: driverId\n"
        "  columns: [driverId, nationality]\n"
        "results:\n"
        "  pkey: resultId\n"
        "  time_col: date\n"
        "  fkeys: {driverId: drivers}\n"
    )
    spec = DatabaseSpec.from_yaml(str(tmp_path / "db.yaml"), data_dir=str(tmp_path))
    assert set(spec.tables) == {"drivers", "results"}
    assert spec.tables["drivers"].columns == ["driverId", "nationality"]
    assert spec.tables["results"].fkeys == {"driverId": "drivers"}
    assert spec.tables["drivers"].path == str(tmp_path / "drivers.parquet")


def test__database_spec__from_yaml__unknown_table_field__raises(tmp_path: Path) -> None:
    """An unexpected key in a table entry is rejected on load with a clear error."""
    (tmp_path / "db.yaml").write_text("drivers:\n  pkey: driverId\n  time_column: t\n")
    with pytest.raises(ValueError, match="Invalid database YAML"):
        DatabaseSpec.from_yaml(str(tmp_path / "db.yaml"))
