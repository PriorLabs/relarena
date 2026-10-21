"""Test discovery and the supplied-split boundary without a RelFlow runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.relflow import RelFlowSystem
from relarena.models.relflow import model as model_module
from relarena_core.dataset import InnerSplit, OuterSplit
from relarena_core.identity import RunIdentity
from relarena_core.registry import registry


@pytest.fixture
def protocol() -> SimpleNamespace:
    """Use distinct censored states and deliberately non-sorted outer anchors."""
    train = SimpleNamespace(df=pd.DataFrame({"entity": [2, 1], "answer": [0, 1]}))
    validation = SimpleNamespace(df=pd.DataFrame({"entity": [1, 2], "answer": [1, 0]}))
    masked = SimpleNamespace(df=pd.DataFrame({"entity": [9, 3, 9]}))
    inner = InnerSplit(
        db_state=object(),
        cutoff=pd.Timestamp("2020-01-01"),
        train_table=train,
        eval_table=validation,
        eval_target=validation,
    )
    outer = OuterSplit(
        db_state=object(),
        cutoff=pd.Timestamp("2021-01-01"),
        train_table=train,
        val_table=validation,
        eval_table=masked,
    )
    contract = {
        "entity_table": "entities",
        "anchor_entity_column": "entity",
        "timestamp_column": "date",
        "target_column": "answer",
        "task_type": "binary_classification",
        "sources": {"entities": {"columns": ["entity"]}},
    }
    task = SimpleNamespace(
        entity_table="entities",
        entity_col="entity",
        time_col="date",
        target_col="answer",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    return SimpleNamespace(
        task=task,
        inner=inner,
        outer=outer,
        recipe={"contract": contract, "query": "SELECT * FROM anchors"},
        identity=RunIdentity("rel-f1", "data-sha", "driver-dnf", "task-sha"),
    )


def test_registered_as_versioned_native_system() -> None:
    assert registry.get("relflow-2026-10-21") is RelFlowSystem
    assert registry.get("relflow-latest") is RelFlowSystem
    assert registry.kind("relflow-latest") == "system"
    with pytest.raises(TypeError, match="no harness search space"):
        registry.search_space("relflow-latest")
    assert RelFlowSystem.supported_task_types == {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
    }


def test_discovery_does_not_import_optional_backend() -> None:
    script = """
import builtins
import sys

original = builtins.__import__
blocked = {'relflow', 'lightning', 'transformers', 'sentence_transformers'}

def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Optional backend imported during discovery: ' + name)
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
from relarena_core import discover_models
from relarena_core.registry import registry
discover_models()
assert registry.kind('relflow-latest') == 'system'
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("time_limit", [None, 60.0])
def test_run_uses_supplied_censored_states_and_only_inner_labels(
    monkeypatch: pytest.MonkeyPatch,
    protocol: SimpleNamespace,
    time_limit: float | None,
) -> None:
    recipe = protocol.recipe
    monkeypatch.setattr(model_module, "load_recipe", lambda dataset, task: recipe)
    data = ModuleType("relarena.models.relflow.data")
    training = ModuleType("relarena.models.relflow.training")
    calls = []
    trained_model = object()

    def snapshot(db: object, directory: Path, *, sources: dict) -> tuple[object, Path]:
        assert sources is recipe["contract"]["sources"]
        calls.append(("snapshot", db))
        return (db, directory)

    def prepare(
        snapshot: tuple[object, Path],
        table: Any,
        task: Any,
        query: str,
        directory: Path,
        *,
        include_target: bool,
    ) -> Path:
        calls.append(("prepare", snapshot[0], table, include_target))
        assert task is protocol.task
        assert query == recipe["query"]
        return directory.with_suffix(".parquet")

    def train(
        train_path: Path,
        validation_path: Path,
        supplied_recipe: dict,
        directory: Path,
        *,
        seed: int,
        deadline: float,
    ) -> object:
        assert train_path.name == "train.parquet"
        assert validation_path.name == "validate.parquet"
        assert supplied_recipe is recipe
        assert seed == 17
        assert np.isfinite(deadline)
        calls.append(("train",))
        return trained_model

    def predict(model: object, path: Path, *, seed: int) -> np.ndarray:
        assert model is trained_model
        assert path.name == "predict.parquet"
        assert seed == 17
        calls.append(("predict",))
        return np.array([0.9, 0.3, 0.9])

    data.snapshot, data.prepare = snapshot, prepare
    training.train, training.predict = train, predict
    monkeypatch.setitem(sys.modules, data.__name__, data)
    monkeypatch.setitem(sys.modules, training.__name__, training)
    predictions = RelFlowSystem(run_identity=protocol.identity).run(
        protocol.task,
        inner_split=protocol.inner,
        outer_split=protocol.outer,
        seed=17,
        time_limit=time_limit,
    )
    np.testing.assert_array_equal(predictions, [0.9, 0.3, 0.9])
    assert calls == [
        ("snapshot", protocol.inner.db_state),
        ("prepare", protocol.inner.db_state, protocol.inner.train_table, True),
        ("prepare", protocol.inner.db_state, protocol.inner.eval_table, True),
        ("train",),
        ("snapshot", protocol.outer.db_state),
        ("prepare", protocol.outer.db_state, protocol.outer.eval_table, False),
        ("predict",),
    ]


def test_run_requires_explicit_task_identity(protocol: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="RunIdentity"):
        RelFlowSystem().run(
            protocol.task,
            inner_split=protocol.inner,
            outer_split=protocol.outer,
            seed=0,
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("entity_table", "other"),
        ("entity_col", "other"),
        ("time_col", "other"),
        ("target_col", "other"),
        ("task_type", TaskType.REGRESSION),
    ],
)
def test_run_rejects_different_task_contract(
    monkeypatch: pytest.MonkeyPatch,
    protocol: SimpleNamespace,
    attribute: str,
    value: Any,
) -> None:
    monkeypatch.setattr(model_module, "load_recipe", lambda *_: protocol.recipe)
    setattr(protocol.task, attribute, value)
    with pytest.raises(ValueError, match="RelFlow recipe"):
        RelFlowSystem(run_identity=protocol.identity).run(
            protocol.task,
            inner_split=protocol.inner,
            outer_split=protocol.outer,
            seed=0,
        )


def test_run_rejects_unmasked_outer_labels(
    monkeypatch: pytest.MonkeyPatch, protocol: SimpleNamespace
) -> None:
    monkeypatch.setattr(model_module, "load_recipe", lambda *_: protocol.recipe)
    protocol.outer.eval_table.df["answer"] = [1, 0, 1]
    with pytest.raises(ValueError, match="masked labels"):
        RelFlowSystem(run_identity=protocol.identity).run(
            protocol.task,
            inner_split=protocol.inner,
            outer_split=protocol.outer,
            seed=0,
        )


@pytest.mark.parametrize("time_limit", [0, -1, float("nan"), float("inf")])
def test_invalid_runtime_budget_fails_before_loading_recipe(
    protocol: SimpleNamespace, time_limit: float
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RelFlowSystem(run_identity=protocol.identity).run(
            protocol.task,
            inner_split=protocol.inner,
            outer_split=protocol.outer,
            seed=0,
            time_limit=time_limit,
        )
