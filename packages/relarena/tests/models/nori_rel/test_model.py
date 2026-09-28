"""Tests for the Nori-Rel RelArena model."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.nori_rel import model as model_module
from relarena.models.nori_rel.model import NoriRel
from relarena_core.featurization.dfs import DFS_MAX_DEPTH
from relarena_core.registry import registry
from relarena_core.search_space import TaskStats, resolve_search_space


class FakeNoriRegressor:
    """Records what the model hands Nori."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, features: pd.DataFrame, target: pd.Series) -> FakeNoriRegressor:
        self.fit_features = features.copy()
        self.target = target
        return self

    def predict(self, features: pd.DataFrame, *, output_type: str) -> np.ndarray:
        self.predict_features = features.copy()
        self.output_type = output_type
        return np.resize(np.array([-0.25, 0.5, 1.25]), len(features))


def _install_fake_nori(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_module, "_load_nori", lambda: (FakeNoriRegressor, nullcontext)
    )
    monkeypatch.setattr(model_module, "_checkpoint_path", lambda: "/nori-30m.pt")


def _features_of(frame: pd.DataFrame, categorical: list[str] | None = None) -> Any:
    """A `build_dfs_features` stand-in returning `frame` sized to the table."""
    calls: list[dict[str, Any]] = []

    def build(
        task: Any, db: Any, table: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db
        calls.append(kwargs)
        return frame.iloc[: len(table.df)].reset_index(drop=True), list(
            categorical or []
        )

    build.calls = calls  # type: ignore[attr-defined]
    return build


def _task(task_type: TaskType) -> SimpleNamespace:
    return SimpleNamespace(target_col="target", time_col=None, task_type=task_type)


def test_registration_and_single_configuration() -> None:
    assert registry.get("nori-rel") is NoriRel
    assert NoriRel.supported_task_types == {
        TaskType.REGRESSION,
        TaskType.BINARY_CLASSIFICATION,
    }
    space = resolve_search_space(
        registry.search_space_for(NoriRel), TaskStats(num_train_nodes=3)
    )
    assert space.configs(10, seed=0) == [{}]


def test_import_does_not_load_optional_nori_dependency() -> None:
    code = (
        "import sys; import relarena.models.nori_rel; "
        "assert 'synthefy_nori' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("task_type", "output_type", "expected"),
    [
        (TaskType.REGRESSION, "median", [-0.25, 0.5, 1.25]),
        (TaskType.BINARY_CLASSIFICATION, "mean", [0.0, 0.5, 1.0]),
    ],
)
def test_prediction_contract(
    monkeypatch: pytest.MonkeyPatch,
    task_type: TaskType,
    output_type: str,
    expected: list[float],
) -> None:
    """Regression returns the median; classification a clipped mean risk score."""
    build = _features_of(
        pd.DataFrame({"value": np.arange(3, dtype=float), "group": ["x"] * 3}),
        ["group"],
    )
    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", build)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    task = _task(task_type)
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(3)))
    model = NoriRel({})

    model.fit(task, object(), train, None, seed=7)
    prediction = model.predict(task, object(), query)

    assert model._model.kwargs == {
        "model_path": "/nori-30m.pt",
        "categorical_columns": ["group"],
        "memory_policy": model_module._memory_policy(),
        "large_context_policy": "random",
        "large_context_threshold": 4,
        "large_context_seed": 7,
    }
    assert model._model.output_type == output_type
    np.testing.assert_array_equal(prediction, expected)


def test_features_slice_the_shared_dfs_matrix_to_depth_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache key follows max_depth, so read the shared deepest matrix."""
    build = _features_of(pd.DataFrame({"value": np.arange(3, dtype=float)}))
    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", build)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(3)))
    model = NoriRel({})

    model.fit(_task(TaskType.REGRESSION), object(), train, None, seed=0)
    model.predict(_task(TaskType.REGRESSION), object(), query)

    assert [(call["depth"], call["max_depth"]) for call in build.calls] == [
        (2, DFS_MAX_DEPTH),
        (2, DFS_MAX_DEPTH),
    ]


def test_all_missing_training_columns_leave_fit_and_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column empty in the training context reaches neither fit nor predict."""

    def build(
        task: Any, db: Any, table: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db, kwargs
        rows = len(table.df)
        sparse = [None] * rows if rows == 3 else ["seen"] * rows
        return pd.DataFrame(
            {"value": np.arange(rows, dtype=float), "sparse": sparse}
        ), ["sparse"]

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", build)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(2)))
    model = NoriRel({})

    model.fit(_task(TaskType.BINARY_CLASSIFICATION), object(), train, None, seed=0)
    model.predict(_task(TaskType.BINARY_CLASSIFICATION), object(), query)

    assert model._columns == ["value"]
    assert model._model.kwargs["categorical_columns"] == []
    assert list(model._model.predict_features) == ["value"]


def test_text_column_empty_in_training_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A text column made only of stringified nulls is empty, not prose."""

    def attach(
        features: pd.DataFrame, db: Any, task: Any, split: Any, columns: Any
    ) -> tuple[pd.DataFrame, list[str]]:
        del db, task, split, columns
        output = features.copy()
        output["notes__raw_text"] = ["nan", "None", "<NA>"][: len(output)]
        return output, ["notes__raw_text"]

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(
        model_module,
        "build_dfs_features",
        _features_of(pd.DataFrame({"value": np.arange(3, dtype=float)})),
    )
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: ["notes"])
    monkeypatch.setattr(model_module, "attach_anchor_text", attach)
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    model = NoriRel({})

    model.fit(_task(TaskType.BINARY_CLASSIFICATION), object(), train, None, seed=0)

    assert model._columns == ["value"]
    assert "text_columns" not in model._model.kwargs


def test_large_context_uses_seeded_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(
        model_module,
        "build_dfs_features",
        _features_of(pd.DataFrame({"a": range(3), "b": range(3)})),
    )
    monkeypatch.setattr(model_module, "CONTEXT_ELEMENTS_BUDGET", 3)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    train = SimpleNamespace(df=pd.DataFrame({"target": [1.0, 2.0, 3.0]}))
    model = NoriRel({})

    model.fit(_task(TaskType.REGRESSION), object(), train, None, seed=0)

    assert model._model.kwargs["large_context_threshold"] == 1
    assert model._model.kwargs["large_context_policy"] == "random"


def test_text_context_is_seeded_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "MAX_TEXT_CONTEXT_ROWS", 3)

    np.testing.assert_array_equal(
        model_module._text_context_rows(8, 7),
        np.random.default_rng(7).permutation(8)[:3],
    )
    np.testing.assert_array_equal(model_module._text_context_rows(3, 7), [0, 1, 2])


def test_memory_policy_is_exact_and_reads_the_gpu_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(model_module.GPU_BUDGET_GB_ENV, "20")

    assert model_module._memory_policy() == {
        "allow_quantization": False,
        "allow_subsample": False,
        "context_row_chunk": None,
        "elements_budget": model_module.CONTEXT_ELEMENTS_BUDGET,
        "gpu_budget_absolute_gb": 20.0,
    }


def test_public_checkpoint_is_pinned_and_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nori.pt"
    path.write_bytes(b"released weights")
    monkeypatch.setattr(model_module, "_download_checkpoint", lambda: str(path))
    monkeypatch.setattr(
        model_module,
        "CHECKPOINT_SHA256",
        hashlib.sha256(b"released weights").hexdigest(),
    )
    model_module._checkpoint_path.cache_clear()
    model_module._sha256.cache_clear()

    assert model_module._checkpoint_path() == str(path)

    path.write_bytes(b"different weights")
    model_module._checkpoint_path.cache_clear()
    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        model_module._checkpoint_path()
