from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from relbench.base import Database, Dataset, EntityTask, Table, TaskType

from relarena.models.relflow.data import prepare, snapshot

pytest.importorskip("duckdb")


class SyntheticDataset(Dataset):
    val_timestamp = pd.Timestamp("2021-01-04")
    test_timestamp = pd.Timestamp("2022-01-01")

    def make_db(self) -> Database:
        raise AssertionError("Materialization must not load the dataset")


class SyntheticTask(EntityTask):
    entity_col = "uid"
    entity_table = "users"
    time_col = "date"
    target_col = "label"
    task_type = TaskType.REGRESSION
    timedelta = pd.Timedelta(days=1)


QUERY = """
WITH anchors AS (
    SELECT * FROM read_parquet($anchors)
    WHERE __row_position >= $start AND __row_position < $stop
)
SELECT anchor.__row_position,
       list(struct_pack(amount := CAST(event.amount AS DECIMAL(10, 2)))
            ORDER BY event.__source_row_id) AS history
FROM anchors AS anchor
LEFT JOIN events AS event
    ON anchor.__entity_key = event.uid AND event.date <= anchor.__as_of
GROUP BY anchor.__row_position
ORDER BY anchor.__row_position DESC
"""


@pytest.fixture
def supplied() -> tuple[Database, Table, EntityTask]:
    users = Table(
        df=pd.DataFrame({"uid": [0, 1], "unused": ["a", "b"]}),
        fkey_col_to_pkey_table={},
        pkey_col="uid",
        time_col=None,
    )
    events = Table(
        df=pd.DataFrame(
            {
                "uid": [0, 0, 1],
                "date": pd.to_datetime(["2020-12-31", "2021-01-03", "2020-12-30"]),
                "amount": [2.5, 99.0, 7.25],
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="date",
    )
    anchors = Table(
        df=pd.DataFrame(
            {
                "uid": [1, 0, 0],
                "date": pd.to_datetime(["2021-01-02", "2021-01-02", "2021-01-01"]),
                "label": [4.0, 8.0, 1.0],
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="date",
    )
    task = SyntheticTask(SyntheticDataset())
    return Database({"users": users, "events": events}), anchors, task


def test_order_temporal_filter_and_nested_decimal(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path
) -> None:
    db, anchors, task = supplied
    source = snapshot(db, tmp_path / "source")
    path = prepare(
        source,
        anchors,
        task,
        QUERY,
        tmp_path / "train",
        include_target=True,
        fetch_size=2,
    )
    result = pq.read_table(path)
    assert result["__row_position"].to_pylist() == [0, 1, 2]
    assert result["target"].to_pylist() == [4.0, 8.0, 1.0]
    assert result["history"].to_pylist() == [
        [{"amount": 7.25}],
        [{"amount": 2.5}],
        [{"amount": 2.5}],
    ]
    assert result.schema.field("history").type.value_type.field("amount").type == (
        pa.float64()
    )
    assert "__target" not in pq.read_schema(path.parent / "anchors.parquet").names
    assert "label" not in pq.read_schema(path.parent / "anchors.parquet").names


def test_masked_outer_table_and_separate_phase_snapshots(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path
) -> None:
    db, anchors, task = supplied
    masked = Table(
        df=anchors.df.drop(columns="label"),
        fkey_col_to_pkey_table=anchors.fkey_col_to_pkey_table,
        pkey_col=None,
        time_col="date",
    )
    inner = snapshot(db, tmp_path / "inner")
    outer_db = Database(
        {
            **db.table_dict,
            "events": Table(
                df=db.table_dict["events"].df.assign(amount=[12.5, 99.0, 17.25]),
                fkey_col_to_pkey_table={"uid": "users"},
                pkey_col=None,
                time_col="date",
            ),
        }
    )
    outer = snapshot(outer_db, tmp_path / "outer")
    result = pq.read_table(
        prepare(outer, masked, task, QUERY, tmp_path / "predict", include_target=False)
    )
    assert "target" not in result.column_names
    assert result["history"].to_pylist()[0] == [{"amount": 17.25}]
    inner_result = pq.read_table(
        prepare(
            inner, anchors, task, QUERY, tmp_path / "inner-train", include_target=True
        )
    )
    assert inner_result["history"].to_pylist()[0] == [{"amount": 7.25}]
    assert not (tmp_path / "predict" / "labels.parquet").exists()
    with pytest.raises(ValueError, match="must be masked"):
        prepare(outer, anchors, task, QUERY, tmp_path / "leak", include_target=False)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM read_parquet($anchors) "
        "WHERE __row_position >= $start AND __row_position < $stop "
        "AND __row_position <> 1",
        "SELECT anchor.* FROM read_parquet($anchors) AS anchor "
        "CROSS JOIN range(2) WHERE __row_position >= $start "
        "AND __row_position < $stop",
    ],
)
def test_rejects_missing_or_duplicate_anchor_rows(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path, query: str
) -> None:
    db, anchors, task = supplied
    source = snapshot(db, tmp_path / "source")
    with pytest.raises(ValueError, match="exactly one"):
        prepare(source, anchors, task, query, tmp_path / "bad", include_target=True)
    assert not (tmp_path / "bad" / "observations.parquet").exists()


def test_projection_indexes_and_missing_unused_columns(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path
) -> None:
    import duckdb

    db, anchors, task = supplied
    source = snapshot(
        db,
        tmp_path / "source",
        sources={"users": {"columns": ["uid"], "indexes": [["uid"]]}},
    )
    with duckdb.connect(str(source), read_only=True) as connection:
        assert connection.sql("SELECT * FROM users").columns == [
            "uid",
            "__source_row_id",
        ]
        assert connection.sql("SELECT __source_row_id FROM users").fetchall() == [
            (0,),
            (1,),
        ]
        assert connection.sql("SELECT count(*) FROM duckdb_indexes()").fetchone() == (
            1,
        )
    missing = snapshot(
        db,
        tmp_path / "missing",
        sources={"users": {"columns": ["uid", "absent"], "indexes": [["absent"]]}},
    )
    query = (
        "SELECT anchor.__row_position, users.uid FROM read_parquet($anchors) "
        "AS anchor LEFT JOIN users ON anchor.__entity_key = users.uid "
        "WHERE anchor.__row_position >= $start AND anchor.__row_position < $stop"
    )
    result = pq.read_table(
        prepare(
            missing, anchors, task, query, tmp_path / "available", include_target=True
        )
    )
    assert result["uid"].to_pylist() == [1, 0, 0]
    with pytest.raises(duckdb.BinderException, match="absent"):
        prepare(
            missing,
            anchors,
            task,
            query.replace("users.uid FROM", "users.absent FROM"),
            tmp_path / "required",
            include_target=True,
        )


def test_nonfinite_labels_rejected(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path
) -> None:
    db, anchors, task = supplied
    source = snapshot(db, tmp_path / "source")
    anchors.df.loc[0, "label"] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        prepare(source, anchors, task, QUERY, tmp_path / "bad", include_target=True)


def test_explicit_audited_null_columns_and_present_values(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path
) -> None:
    import duckdb

    db, _, _ = supplied
    sources = {
        "users": {
            "columns": ["uid", "all_null"],
            "indexes": [],
            "nullable_columns": {"all_null": "VARCHAR"},
        }
    }
    source = snapshot(db, tmp_path / "missing", sources=sources)
    with duckdb.connect(str(source), read_only=True) as connection:
        values = connection.sql("SELECT all_null FROM users").to_arrow_table()
        assert values["all_null"].to_pylist() == [None, None]
        assert values.schema.field("all_null").type == pa.string()
    db.table_dict["users"].df["all_null"] = ["present", "retained"]
    source = snapshot(db, tmp_path / "present", sources=sources)
    with duckdb.connect(str(source), read_only=True) as connection:
        assert connection.sql("SELECT all_null FROM users").fetchall() == [
            ("present",),
            ("retained",),
        ]


def test_rejects_query_targets_and_multiple_statements(
    supplied: tuple[Database, Table, EntityTask], tmp_path: Path
) -> None:
    db, anchors, task = supplied
    source = snapshot(db, tmp_path / "source")
    with pytest.raises(ValueError, match="no targets"):
        prepare(
            source,
            anchors,
            task,
            "SELECT __row_position, 0 AS target FROM read_parquet($anchors) "
            "WHERE __row_position >= $start AND __row_position < $stop",
            tmp_path / "target",
            include_target=True,
        )
    with pytest.raises(ValueError, match="exactly one SELECT"):
        prepare(
            source,
            anchors,
            task,
            "SELECT 1; SELECT 2",
            tmp_path / "statements",
            include_target=True,
        )
