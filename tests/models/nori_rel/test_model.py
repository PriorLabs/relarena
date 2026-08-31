"""Unit tests for the Nori-Rel model wrapper."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.nori_rel import model as model_module
from relarena.models.nori_rel.model import NORI_REL_SPACE, NoriRelModel
from relarena.registry import registry


def test__model__registered_with_fixed_regression_scope() -> None:
    assert registry.get("nori-rel") is NoriRelModel
    assert registry.search_space("nori-rel") is NORI_REL_SPACE
    assert NORI_REL_SPACE.configs(10, seed=0) == [{"max_depth": 2}]
    assert NoriRelModel.supported_task_types == frozenset({TaskType.REGRESSION})


def test__module_import__does_not_load_optional_nori_dependency() -> None:
    code = (
        "import sys; import relarena.models.nori_rel; "
        "assert 'synthefy_nori' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test__checkpoint__pinned_and_sha_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nori.pt"
    path.write_bytes(b"released weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = 0

    def fake_download() -> str:
        nonlocal calls
        calls += 1
        return str(path)

    assert model_module.CHECKPOINT_SHA256 == (
        "818433f8af12c1137b96d9ff47e109b4eef5818d4e52a9656b2e573dbf13b74d"
    )
    monkeypatch.setattr(model_module, "_download_checkpoint", fake_download)
    monkeypatch.setattr(model_module, "CHECKPOINT_SHA256", digest)
    model_module._checkpoint_path.cache_clear()
    model_module._sha256.cache_clear()

    assert model_module.NORI_MODEL == "nori-30m"
    assert model_module.CHECKPOINT_REVISION == (
        "63c9f7facf9fb32c37ce3fc2fba331d524696318"
    )
    assert model_module._checkpoint_path() == str(path)
    assert calls == 1

    path.write_bytes(b"different weights")
    model_module._checkpoint_path.cache_clear()
    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        model_module._checkpoint_path()
    model_module._checkpoint_path.cache_clear()
    model_module._sha256.cache_clear()


def test__fit_predict__uses_depth_two_and_median_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_calls: list[dict[str, object]] = []

    class FakeNoriRegressor:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            self.fit_rows = len(features)
            self.target = target

        def predict(self, features: pd.DataFrame, *, output_type: str) -> np.ndarray:
            self.output_type = output_type
            return np.array([-0.25, 0.5, 1.25])

    def fake_features(
        task: object, db: object, table: object, **kwargs: object
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db
        feature_calls.append(kwargs)
        rows = len(table.df)  # type: ignore[attr-defined]
        return pd.DataFrame({"value": np.arange(rows), "group": ["x"] * rows}), [
            "group"
        ]

    monkeypatch.setattr(
        model_module, "_load_nori", lambda: (FakeNoriRegressor, nullcontext)
    )
    monkeypatch.setattr(model_module, "build_dfs_features", fake_features)
    monkeypatch.setattr(model_module, "_checkpoint_path", lambda: "/nori-30m.pt")

    task = SimpleNamespace(target_col="target", time_col="timestamp")
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 2.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(3)))
    model = NoriRelModel({"max_depth": 2})

    model.fit(task, object(), train, None, seed=7)
    prediction = model.predict(task, object(), query)

    assert model._model.kwargs == {
        "model_path": "/nori-30m.pt",
        "categorical_columns": ["group"],
        "memory_policy": model_module._memory_policy(),
        "large_context_policy": model_module._random_window,
        "large_context_threshold": 4,
        "large_context_seed": 7,
    }
    assert model._model.fit_rows == 3
    assert model._model.output_type == "median"
    assert [call["depth"] for call in feature_calls] == [2, 2]
    assert [call["max_depth"] for call in feature_calls] == [2, 2]
    np.testing.assert_array_equal(prediction, [-0.25, 0.5, 1.25])


def test__large_context__activates_seeded_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNoriRegressor:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            del features, target

    def fake_features(
        *args: object, **kwargs: object
    ) -> tuple[pd.DataFrame, list[str]]:
        del args, kwargs
        return pd.DataFrame({"a": range(3), "b": range(3)}), []

    monkeypatch.setattr(
        model_module, "_load_nori", lambda: (FakeNoriRegressor, nullcontext)
    )
    monkeypatch.setattr(model_module, "build_dfs_features", fake_features)
    monkeypatch.setattr(model_module, "_checkpoint_path", lambda: "/nori-30m.pt")
    monkeypatch.setattr(model_module, "_CONTEXT_ELEMENTS", 3)

    task = SimpleNamespace(target_col="target", time_col=None)
    train = SimpleNamespace(df=pd.DataFrame({"target": [1.0, 2.0, 3.0]}))
    model = NoriRelModel({"max_depth": 2})
    model.fit(task, object(), train, None, seed=0)

    assert model._model.kwargs["large_context_threshold"] == 1
    assert model._model.kwargs["large_context_policy"] is model_module._random_window


def test__random_window__forces_cache_safe_query_path() -> None:
    class Problem:
        n_train = 8
        window = 3
        n_test = 2
        query_chunk = 25_000

        def predict(self, pool: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
            self.pool = pool
            self.query = query_idx
            return query_idx

    problem = Problem()
    expected = np.random.default_rng(7).permutation(problem.n_train)[: problem.window]
    prediction = model_module._random_window(problem, np.random.default_rng(7))

    assert problem.query_chunk == 5
    np.testing.assert_array_equal(problem.pool, expected)
    np.testing.assert_array_equal(problem.query, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(prediction, [0, 1])


@pytest.mark.parametrize("n_test", [2, 50_000])
def test__random_window__caps_each_forward_pass(n_test: int) -> None:
    class Problem:
        n_train = 40_001
        window = 40_000
        query_chunk = 25_000

        def __init__(self) -> None:
            self.n_test = n_test

        def predict(self, pool: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
            self.pool = pool
            self.query = query_idx
            self.forward_rows = [
                len(pool) + len(query_idx[start : start + self.query_chunk])
                for start in range(0, len(query_idx), self.query_chunk)
            ]
            return query_idx

    problem = Problem()
    model_module._random_window(problem, np.random.default_rng(0))

    assert len(problem.pool) + problem.query_chunk == model_module._MAX_FORWARD_ROWS
    assert max(problem.forward_rows) == model_module._MAX_FORWARD_ROWS


def test__config__rejects_nonreported_depth() -> None:
    model = NoriRelModel({"max_depth": 3})
    task = SimpleNamespace(target_col="target", time_col=None)
    train = SimpleNamespace(df=pd.DataFrame({"target": [1.0]}))

    with pytest.raises(ValueError, match="requires max_depth=2"):
        model.fit(task, object(), train, None, seed=0)
