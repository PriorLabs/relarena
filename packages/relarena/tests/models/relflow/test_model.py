from __future__ import annotations

import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table, TaskType

from relarena.models.relflow import RelFlowModel, search_space
from relarena_core.registry import registry
from relarena_core.search_space import TaskStats


def fixture(
    task_type: TaskType = TaskType.REGRESSION,
) -> tuple[SimpleNamespace, Database, Table]:
    dataset = SimpleNamespace(val_timestamp=pd.Timestamp("2020-01-10"))
    task = SimpleNamespace(
        dataset=dataset,
        task_type=task_type,
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
                        "entity_id": [0, 1, 2, 3],
                        "amount": [1.5, 3.0, 5.5, 9.0],
                        "future_only": [None, None, None, 17.0],
                        "created": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-20"]),
                    }
                ),
                {},
                pkey_col="entity_id",
                time_col="created",
            ),
            "events": Table(
                pd.DataFrame(
                    {
                        "event_id": [0, 1, 2, 3],
                        "entity_id": [0, 0, 1, 1],
                        "value": [1.0, None, 3.0, 100.0],
                        "timestamp": pd.to_datetime(
                            ["2020-01-02", "2020-01-04", "2020-01-02", "2020-01-15"]
                        ),
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
                "label": [0, 1, 0, 1, 0, 1]
                if task_type == TaskType.BINARY_CLASSIFICATION
                else [1.2, 2.0, 3.0, 1.4, 2.2, 3.2],
            }
        ),
        {"entity_id": "entities"},
        time_col="timestamp",
    )
    return task, db, anchors


def test__registry__without_optional_backend__registers() -> None:
    assert registry.get(RelFlowModel.name) is RelFlowModel
    assert registry.get("relflow-latest") is RelFlowModel
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "relflow" or name.startswith("relflow."):
        raise RuntimeError("Optional backend imported during registration")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from relarena.models.relflow import RelFlowModel
assert RelFlowModel.name == "relflow-2026-11-04"
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ)


@pytest.mark.parametrize(
    "rows,tier,caps",
    [
        (19_239, "small", {"outer": 128, "inner": 32}),
        (20_000, "medium", {"outer": 64, "inner": 32}),
        (499_999, "medium", {"outer": 64, "inner": 32}),
        (500_000, "large", {"outer": 32, "inner": 16}),
    ],
)
def test__space__original_training_rows__freezes_single_default(
    rows: int, tier: str, caps: dict[str, int]
) -> None:
    space = search_space(TaskStats(rows))
    assert not space.is_tunable
    assert space.configs(n_trials=50, seed=0) == [space.default_overrides]
    assert space.default_overrides["tier"] == tier
    assert space.default_overrides["caps"] == caps
    assert space.default_overrides["execution"]["duckdb_memory_limit"] == (
        "32GB" if tier == "large" else "16GB"
    )
    assert space.default_overrides["recipe"]["passes"] == 8
    assert space.default_overrides["recipe"]["overflow"] == "random"


def test__context__outer_database__profiles_only_validation_time(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")
    import pyarrow.parquet as pq

    from relarena.models.relflow import raw
    from relarena.models.relflow.context import Context, infer, materialize

    task, db, anchors = fixture()
    config = search_space(TaskStats(19_239)).default_overrides
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        structure = infer(context, task, config)
        assert "future_only" not in raw.model(structure)["fields"]
        assert "DISTINCT" not in raw.query(structure)
        context.freeze()
        path = materialize(
            context, structure, task, anchors, tmp_path / "data.parquet", labelled=True
        )
        observations = pq.read_table(path)
        assert observations["__row_position"].to_pylist() == list(range(len(anchors)))
        assert observations["target"].to_pylist() == anchors.df["label"].tolist()
        histories = observations["events"].to_pylist()
        assert histories[2] == []
        assert any(item["value"] is None for item in histories[0])
        assert len(histories[4]) == 1  # January 15 event is invisible to stale val.
    finally:
        context.close()


@pytest.mark.skipif(
    importlib.util.find_spec("relflow") is None, reason="Needs relflow extra"
)
@pytest.mark.parametrize("kind", [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION])
def test__fit_predict__tiny_supplied_database__satisfies_contract(
    kind: TaskType,
) -> None:
    import torch

    torch.set_num_threads(1)
    task, db, train = fixture(kind)
    config = search_space(TaskStats(3)).default_overrides
    model = RelFlowModel(config)
    model.fit(task, db, train, None, seed=0)
    assert model.fitted.model.trainer.global_step == 8
    frame = train.df.iloc[[2, 0, 1, 0]].drop(columns=[task.target_col])
    table = Table(frame, train.fkey_col_to_pkey_table, time_col=train.time_col)
    prediction = model.predict(task, db, table)
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert prediction[1] == prediction[3]
    assert np.array_equal(prediction, model.predict(task, db, table))
    if kind == TaskType.BINARY_CLASSIFICATION:
        assert np.all((prediction >= 0) & (prediction <= 1))
    empty = Table(frame.iloc[:0], table.fkey_col_to_pkey_table, time_col=table.time_col)
    assert model.predict(task, db, empty).shape == (0,)


def test__context__pandas_categories__retains_string_fields(tmp_path: Path) -> None:
    from relarena.models.relflow import raw
    from relarena.models.relflow.context import Context, infer

    task, db, _ = fixture()
    db.table_dict["entities"].df["segment"] = pd.Categorical(["a", "b", None, "c"])
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        structure = infer(context, task, search_space(TaskStats(3)).default_overrides)
        assert raw.model(structure)["fields"]["segment"]["type"] == "hash"
    finally:
        context.close()


def test__context__timestamp_ties__uses_stable_input_ordinals(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from relarena.models.relflow.context import Context, infer, materialize

    task, db, anchors = fixture()
    db.table_dict["events"].df.loc[1, "timestamp"] = pd.Timestamp("2020-01-02")
    config = search_space(TaskStats(3)).default_overrides
    observed = []
    for index in range(2):
        context = Context.open(db, task.dataset.val_timestamp, tmp_path)
        try:
            structure = infer(context, task, config)
            path = materialize(
                context,
                structure,
                task,
                anchors,
                tmp_path / f"ties-{index}.parquet",
                labelled=True,
            )
            observed.append(pq.read_table(path)["events"].to_pylist()[0])
        finally:
            context.close()
    assert observed[0] == observed[1]
    assert [row["value"] for row in observed[0]] == [1.0, None]
    assert "__source_row_id" not in db.table_dict["events"].df


@pytest.mark.parametrize(
    "rows,dataset,task,expected",
    [
        (19_239, "rel-event", "user-attendance", 64),
        (59_454, "rel-avito", "user-clicks", 512),
        (86_619, "rel-avito", "user-visits", 512),
        (151_407, "rel-trial", "site-success", 512),
        (43_335, "rel-trial", "study-adverse", 512),
        (3_386_276, "rel-stack", "user-badge", 512),
        (50_000, "unseen", "new-task", 512),
    ],
)
def test__fit__task_identity__uses_only_tier_batch(
    rows: int,
    dataset: str,
    task: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    from relarena_core.identity import RunIdentity

    config = search_space(TaskStats(rows)).default_overrides
    backend = SimpleNamespace(fit=Mock())
    monkeypatch.setitem(sys.modules, "relarena.models.relflow.training", backend)
    task_spec, db, anchors = fixture()
    for identity in (RunIdentity(dataset, None, task, None), None):
        model = RelFlowModel(config, run_identity=identity)
        model.fit(task_spec, db, anchors, None, seed=0)
        resolved = backend.fit.call_args.args[3]
        assert resolved["execution"]["batch_size"] == expected
        assert resolved == config
        assert resolved["execution"]["text_resident_mib"] == (
            256 if rows < 500_000 else 2048
        )


def test__execution__large_loader_and_text__retains_bounded_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("relflow")
    from relarena.models.relflow.training import schema

    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        hub, "snapshot_download", lambda **kwargs: str(tmp_path / "encoder")
    )
    task, db, _ = fixture()
    from relarena.models.relflow.context import Context, infer

    db.table_dict["entities"].df["description"] = [
        "A detailed description of a different financial transaction with context.",
        "Another long description of the customer account and recent activity.",
        "A third description of the merchant profile and payment preferences.",
        "Future content that remains outside the schema profiling boundary.",
    ]
    config = search_space(TaskStats(500_000)).default_overrides
    assert config["execution"] == {
        "batch_size": 512,
        "workers": 2,
        "persistent_workers": True,
        "prefetch_factor": 1,
        "shuffle_rows": 2048,
        "text_token_cache": 16384,
        "text_resident_mib": 2048,
        "duckdb_threads": 8,
        "duckdb_memory_limit": "32GB",
        "duckdb_spill_limit": "256GB",
    }
    context = Context.open(db, task.dataset.val_timestamp, tmp_path)
    try:
        structure = infer(context, task, config)
        configured = schema(structure, config, store=tmp_path / "text-store")
        assert configured["fields"]["events"]["length"] == 32
        assert configured["fields"]["events"]["overflow"] == "random"
        text = configured["fields"]["description"]
        assert text["type"] == "text"
        assert text["cache"] == 16384
        assert text["resident"] == 2048
        assert text["store"] == str(tmp_path / "text-store")
    finally:
        context.close()


@pytest.mark.parametrize("predict", [False, True])
def test__loader__parquet_source__restarts_with_archived_scan_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, predict: bool
) -> None:
    pytest.importorskip("relflow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from relarena.models.relflow import training

    path = tmp_path / "records.parquet"
    pq.write_table(pa.table({"ordinal": np.arange(8193)}), path, row_group_size=8193)
    monkeypatch.setattr(
        training.rf, "ArrowDataModule", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    config = search_space(TaskStats(500_000)).default_overrides
    datamodule = training.loader(None, path, 0, config["execution"], predict=predict)
    split = "predict" if predict else "train"
    source = pickle.loads(pickle.dumps(getattr(datamodule, split)))
    for _ in range(2):
        scanned = list(source())
        assert [len(batch) for batch in scanned] == [8192, 1]
        assert pa.Table.from_batches(scanned)["ordinal"].to_pylist() == list(
            range(8193)
        )
    assert datamodule.shuffle == {split: not predict}
    assert datamodule.shuffle_rows == 2048
    assert datamodule.num_workers == 0
