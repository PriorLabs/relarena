"""Adapter boundary checks for fixed-point inputs and cleaned FK metadata."""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from relbench.base import Database, Table, TaskType

from relarena.models.relflow import RelFlowModel, raw, search_space
from relarena.models.relflow.context import Context, infer, materialize
from relarena_core.search_space import TaskStats


def decimal_fixture() -> tuple[SimpleNamespace, Database, Table]:
    task = SimpleNamespace(
        dataset=SimpleNamespace(val_timestamp=pd.Timestamp("2020-01-10")),
        task_type=TaskType.REGRESSION,
        entity_table="entities",
        entity_col="entity_id",
        time_col="timestamp",
        target_col="label",
    )
    db = Database(
        {
            "entities": Table(
                pd.DataFrame(
                    {
                        "entity_id": [0, 1, 2],
                        "amount": [Decimal("1.25"), None, Decimal("7.125")],
                        "count": [2, 4, 6],
                        "flag": [Decimal("1.00"), None, Decimal("1.00")],
                    }
                ),
                {},
                pkey_col="entity_id",
            ),
            "events": Table(
                pd.DataFrame(
                    {
                        "event_id": [0, 1, 2, 3],
                        "entity_id": [0, 0, 1, 1],
                        "value": [
                            Decimal("2.25"),
                            None,
                            Decimal("4.50"),
                            Decimal("6.75"),
                        ],
                        "timestamp": pd.to_datetime(["2020-01-02"] * 4),
                    }
                ),
                {"entity_id": "entities"},
                pkey_col="event_id",
                time_col="timestamp",
            ),
        }
    )
    anchors = Table(
        pd.DataFrame(
            {
                "entity_id": [0, 1, 2, 0, 1, 2],
                "timestamp": pd.to_datetime(["2020-01-05"] * 3 + ["2020-01-12"] * 3),
                "label": [
                    Decimal(value)
                    for value in ("1.2", "2.0", "3.0", "1.4", "2.2", "3.2")
                ],
            }
        ),
        {"entity_id": "entities"},
        time_col="timestamp",
    )
    return task, db, anchors


def test__context__cleaned_null_foreign_key__omits_stale_metadata(
    tmp_path: Path,
) -> None:
    task, db, anchors = decimal_fixture()
    events = db.table_dict["events"]
    # The official cleaner drops this all-null physical column but retains its FK.
    events.fkey_col_to_pkey_table["removed_entity_id"] = "entities"
    original = {name: table.df.copy(deep=True) for name, table in db.table_dict.items()}
    metadata = dict(events.fkey_col_to_pkey_table)
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        assert context.schema["events"]["fkeys"] == {"entity_id": "entities"}
        structure = infer(context, task, search_space(TaskStats(3)).default_overrides)
        materialize(
            context,
            structure,
            task,
            anchors,
            tmp_path / "cleaned.parquet",
            labelled=True,
        )
        assert "removed_entity_id" not in raw.query(structure)
    finally:
        context.close()
    assert events.fkey_col_to_pkey_table == metadata
    for name, frame in original.items():
        pd.testing.assert_frame_equal(db.table_dict[name].df, frame)


def test__materialize__decimal_features_at_every_depth__casts_only_output_atoms(
    tmp_path: Path,
) -> None:
    task, db, anchors = decimal_fixture()
    large_integer = 2**53 + 1
    db.table_dict["entities"].df["count"] = [large_integer, 4, 6]
    db.table_dict["entities"].df["dimension_id"] = [0, 1, 2]
    db.table_dict["events"].df["dimension_id"] = [0, 1, 1, 2]
    db.table_dict["dimensions"] = Table(
        pd.DataFrame(
            {
                "dimension_id": [0, 1, 2],
                "rate": [Decimal("1.125"), None, Decimal("3.375")],
            }
        ),
        {},
        pkey_col="dimension_id",
    )
    db.table_dict["lines"] = Table(
        pd.DataFrame(
            {
                "event_id": [0, 0, 1],
                "price": [Decimal("9.875"), None, Decimal("10.125")],
            }
        ),
        {"event_id": "events"},
    )
    dimension = raw.Lookup(
        "dimension",
        "dimensions",
        '{t}."dimension_id" = {row}."dimension_id"',
        (raw.number("rate", "dimension_rate"),),
    )
    structure = raw.Task(
        dataset="fixture",
        task="decimal",
        task_type="regression",
        entity_table="entities",
        entity_key="entity_id",
        anchor_entity_column="entity_id",
        target_column="label",
        root_time=None,
        columns=(raw.number("amount"), raw.number("count"), raw.category("flag")),
        lookups=(dimension,),
        branches=(
            raw.Branch(
                name="events",
                table="events",
                on='{t}."entity_id" = {parent}."__entity_key"',
                time="timestamp",
                length=8,
                columns=(raw.number("value"),),
                lookups=(dimension,),
                child_key="event_id",
                nested=(
                    raw.Branch(
                        name="lines",
                        table="lines",
                        on='{t}."event_id" = {parent}."__child_key"',
                        time=None,
                        length=8,
                        columns=(raw.number("price"),),
                    ),
                ),
            ),
        ),
    )
    original = {name: table.df.copy(deep=True) for name, table in db.table_dict.items()}
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        path = materialize(
            context,
            structure,
            task,
            anchors,
            tmp_path / "decimals.parquet",
            labelled=True,
        )
        assert (
            context.connection.execute('SELECT typeof("amount") FROM entities LIMIT 1')
            .fetchone()[0]
            .startswith("DECIMAL")
        )
    finally:
        context.close()
    observations = pq.read_table(path)
    for name in ("amount", "dimension_rate", "target"):
        assert observations[name].type == pa.float64()
    assert observations["flag"].type == pa.string()
    assert observations["flag"].to_pylist() == ["1.00", None, "1.00"] * 2
    assert observations["amount"].to_pylist() == [1.25, None, 7.125] * 2
    assert observations["dimension_rate"].to_pylist() == [1.125, None, 3.375] * 2
    assert observations["count"].type == pa.int64()
    assert observations["count"][0].as_py() == large_integer
    events = observations["events"].to_pylist()
    assert events[0][0]["value"] == 2.25
    assert events[0][1]["value"] is None
    assert events[0][1]["dimension_rate"] is None
    assert events[0][0]["lines"] == [{"price": 9.875}, {"price": None}]
    assert events[0][1]["lines"] == [{"price": 10.125}]
    assert events[2] == []
    event_type = observations.schema.field("events").type.value_type
    assert event_type.field("value").type == pa.float64()
    assert event_type.field("dimension_rate").type == pa.float64()
    assert event_type.field("lines").type.value_type.field("price").type == pa.float64()
    assert observations["target"].to_pylist() == pytest.approx(
        [1.2, 2.0, 3.0, 1.4, 2.2, 3.2]
    )
    for name, frame in original.items():
        pd.testing.assert_frame_equal(db.table_dict[name].df, frame)


def test__materialize__large_decimal_categories__retain_distinct_labels(
    tmp_path: Path,
) -> None:
    task, db, anchors = decimal_fixture()
    labels = [Decimal("9007199254740992"), Decimal("9007199254740993"), None]
    db.table_dict["entities"].df["segment_code"] = labels
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        structure = infer(context, task, search_space(TaskStats(3)).default_overrides)
        assert raw.model(structure)["fields"]["segment_code"]["type"] == "category"
        path = materialize(
            context,
            structure,
            task,
            anchors,
            tmp_path / "labels.parquet",
            labelled=True,
        )
    finally:
        context.close()
    categories = pq.read_table(path)["segment_code"]
    assert categories.type == pa.string()
    assert categories.to_pylist() == ["9007199254740992", "9007199254740993", None] * 2
    assert len(set(value for value in categories.to_pylist() if value is not None)) == 2
    assert db.table_dict["entities"].df["segment_code"].tolist() == labels


def test__context__configured_resources__sets_explicit_limits(tmp_path: Path) -> None:
    task, db, _ = decimal_fixture()
    context = Context.open(
        db,
        task.dataset.val_timestamp,
        tmp_path,
        threads=2,
        memory_limit="128MB",
        max_temp_directory_size="256MB",
    )
    try:
        configured = dict(
            context.connection.execute(
                "SELECT name, value FROM duckdb_settings() "
                "WHERE name IN ('threads', 'memory_limit', 'max_temp_directory_size')"
            ).fetchall()
        )
        assert configured == {
            "threads": "2",
            "memory_limit": "122.0 MiB",
            "max_temp_directory_size": "244.1 MiB",
        }
    finally:
        context.close()


@pytest.mark.parametrize("timestamp", ["2020-01-03", "2020-01-04"])
def test__materialize__anchor_cutoff__censors_each_relational_depth(
    tmp_path: Path, timestamp: str
) -> None:
    task, db, anchors = decimal_fixture()
    db.table_dict["entities"].df.loc[1, "amount"] = Decimal("3.00")
    db.table_dict["entities"].df["created"] = pd.to_datetime(
        ["2020-01-02", "2020-01-04", "2020-01-02"]
    )
    db.table_dict["entities"].time_col = "created"
    db.table_dict["events"].df.loc[1, "timestamp"] = pd.Timestamp("2020-01-04")
    db.table_dict["events"].df.loc[1, "value"] = Decimal("99.00")
    db.table_dict["dimensions"] = Table(
        pd.DataFrame(
            {
                "entity_id": [0, 0],
                "timestamp": pd.to_datetime(["2020-01-02", "2020-01-04"]),
                "rate": [10.0, 99.0],
            }
        ),
        {"entity_id": "entities"},
        time_col="timestamp",
    )
    db.table_dict["lines"] = Table(
        pd.DataFrame(
            {
                "event_id": [0, 0],
                "timestamp": pd.to_datetime(["2020-01-02", "2020-01-04"]),
                "price": [20.0, 99.0],
            }
        ),
        {"event_id": "events"},
        time_col="timestamp",
    )
    frame = anchors.df.iloc[:2].copy()
    frame[task.time_col] = pd.Timestamp(timestamp)
    anchors = Table(frame, anchors.fkey_col_to_pkey_table, time_col=task.time_col)
    structure = raw.Task(
        dataset="fixture",
        task="anchor-cutoff",
        task_type="regression",
        entity_table="entities",
        entity_key="entity_id",
        anchor_entity_column="entity_id",
        target_column="label",
        root_time="created",
        columns=(raw.number("amount"),),
        lookups=(
            raw.Lookup(
                "dimension",
                "dimensions",
                '{t}."entity_id" = {row}."entity_id"',
                (raw.number("rate"),),
                time="timestamp",
            ),
        ),
        branches=(
            raw.Branch(
                name="events",
                table="events",
                on='{t}."entity_id" = {parent}."__entity_key"',
                time="timestamp",
                length=8,
                columns=(raw.number("value"),),
                child_key="event_id",
                nested=(
                    raw.Branch(
                        name="lines",
                        table="lines",
                        on='{t}."event_id" = {parent}."__child_key"',
                        time="timestamp",
                        length=8,
                        columns=(raw.number("price"),),
                    ),
                ),
            ),
        ),
    )
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        path = materialize(
            context,
            structure,
            task,
            anchors,
            tmp_path / "cutoff.parquet",
            labelled=True,
        )
    finally:
        context.close()
    observations = pq.read_table(path)
    at_boundary = timestamp == "2020-01-04"
    assert observations["amount"].to_pylist() == [1.25, 3.0 if at_boundary else None]
    assert observations["rate"].to_pylist() == [99.0 if at_boundary else 10.0, None]
    events = observations["events"].to_pylist()[0]
    assert [row["value"] for row in events] == ([2.25, 99.0] if at_boundary else [2.25])
    assert [row["price"] for row in events[0]["lines"]] == (
        [20.0, 99.0] if at_boundary else [20.0]
    )


@pytest.mark.skipif(
    importlib.util.find_spec("relflow") is None, reason="Needs relflow extra"
)
def test__fit_predict__derived_decimal_features_and_target__satisfies_contract() -> (
    None
):
    import torch

    torch.set_num_threads(1)
    task, db, train = decimal_fixture()
    original = {name: table.df.copy(deep=True) for name, table in db.table_dict.items()}
    config = search_space(TaskStats(3)).default_overrides
    model = RelFlowModel(config)
    model.fit(task, db, train, None, seed=0)
    assert model.fitted.model.trainer.global_step == 8
    prepared = pq.read_table(Path(model.fitted.directory.name) / "train.parquet")
    assert prepared["amount"].type == pa.float64()
    assert prepared["amount"].null_count == 2
    assert prepared["target"].type == pa.float64()
    assert (
        prepared.schema.field("events").type.value_type.field("value").type
        == pa.float64()
    )
    frame = train.df.iloc[[2, 0, 1, 0]].drop(columns=[task.target_col])
    evaluate = Table(frame, train.fkey_col_to_pkey_table, time_col=train.time_col)
    predictions = model.predict(task, db, evaluate)
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert predictions[1] == predictions[3]
    assert np.array_equal(predictions, model.predict(task, db, evaluate))
    for name, data in original.items():
        pd.testing.assert_frame_equal(db.table_dict[name].df, data)
