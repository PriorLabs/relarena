"""Unit tests for the RDBLearn (DFS + TFM) model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table, TaskType

from relarena.cache import CacheConfig
from relarena.models._shared.tfm import tfm
from relarena.models.rdblearn import RDBLEARN_SPACE, RDBLearnModel
from relarena.models.rdblearn import model as rdblearn
from relarena.registry import registry


def test_registered_under_name_rdblearn() -> None:
    assert "rdblearn" in registry
    assert registry.get("rdblearn") is RDBLearnModel
    assert registry.search_space("rdblearn") is RDBLEARN_SPACE


def test_default_is_first_tfm_at_depth_2() -> None:
    assert RDBLEARN_SPACE.default_overrides == {"tfm": "tabpfn-v2", "max_depth": 2}


def test_supports_binary_and_regression() -> None:
    # Inherits the base-class scope (ENTITY_TASK_TYPES): regression + binary.
    assert RDBLearnModel.supported_task_types == frozenset(
        {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.REGRESSION,
        }
    )


def test_search_grid_is_tfm_by_depth_default_first() -> None:
    grid = RDBLEARN_SPACE.fixed_grid
    tfms = ("tabpfn-v2", "tabpfn-v2.5")
    expected = [{"tfm": t, "max_depth": d} for d in range(2, 5) for t in tfms]
    assert grid == expected
    # full TFM × depth grid; shallowest (the default depth) first so the default
    # regime survives budget truncation in SearchSpace.configs
    assert len(grid) == len(tfms) * 3  # 2 TFMs × depths 2, 3, 4
    assert grid[0]["max_depth"] == 2
    assert grid[0] == RDBLEARN_SPACE.default_overrides
    assert RDBLEARN_SPACE.default_overrides in grid
    # every config names a registered TFM
    assert {c["tfm"] for c in grid} <= set(tfm.TFM_REGISTRY)


def test_prediction_batching_is_configured_by_rdblearn_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_module = ModuleType("tabpfn.settings")
    settings_module.settings = SimpleNamespace(  # type: ignore[attr-defined]
        tabpfn=SimpleNamespace(max_batched_test_rows=32768)
    )
    monkeypatch.setitem(sys.modules, "tabpfn.settings", settings_module)
    monkeypatch.delenv("TABPFN_MAX_BATCHED_TEST_ROWS", raising=False)

    limit = rdblearn._configure_prediction_batching()

    assert limit == 8192
    assert os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] == "8192"
    assert settings_module.settings.tabpfn.max_batched_test_rows == 8192


# -- full-anchor DFS, post-DFS TFM cap ---------------------------------------


class _CaptureRegressor:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_CaptureRegressor":
        self.X = X
        self.y = y
        return self


def test__fit__train_above_cap__featurizes_full_table_then_caps_aligned_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cap, n, seed = 10, 30, 42
    train = Table(
        df=pd.DataFrame(
            {
                "id": np.arange(n),
                "ts": pd.date_range("2020-01-01", periods=n),
                "y": np.arange(float(n)),
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col=None,
        time_col="ts",
    )
    seen: dict[str, object] = {}

    def fake_build(
        task: object,
        db: object,
        table: Table,
        *,
        depth: int,
        max_depth: int,
        history_table: Table | None = None,
        keep_anchor_columns: bool = False,
        cache: CacheConfig,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, list[str]]:
        seen["n_rows"] = len(table.df)
        seen["history_table"] = history_table
        seen["keep_anchor_columns"] = keep_anchor_columns
        seen["cache"] = cache
        return pd.DataFrame({"f": table.df["y"].to_numpy() * 2}), []

    monkeypatch.setattr(rdblearn, "build_dfs_features", fake_build)
    monkeypatch.setitem(
        tfm.TFM_REGISTRY,
        "capture",
        tfm.TFMSpec(
            make_classifier=lambda **kwargs: _CaptureRegressor(),
            make_regressor=lambda **kwargs: _CaptureRegressor(),
            max_train_samples=cap,
        ),
    )
    task = SimpleNamespace(target_col="y", task_type=TaskType.REGRESSION, time_col="ts")
    model = RDBLearnModel(
        config={"tfm": "capture"}, cache=CacheConfig(tmp_path, "raise")
    )
    model.fit(task, db=None, train_table=train, val_table=None, seed=seed)

    assert seen["n_rows"] == n
    assert seen["history_table"] is train
    assert seen["keep_anchor_columns"] is True
    assert seen["cache"] == CacheConfig(tmp_path, "raise")
    idx = tfm._downsample_indices(
        train.df["y"].to_numpy(),
        TaskType.REGRESSION,
        cap,
        np.random.default_rng(seed),
    )
    estimator = model._fitted.estimator
    assert estimator.X["f"].tolist() == (train.df.iloc[idx]["y"] * 2).tolist()
    assert estimator.y.tolist() == train.df.iloc[idx]["y"].tolist()
