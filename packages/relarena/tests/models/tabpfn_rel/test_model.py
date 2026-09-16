"""Wiring tests for `TabPFNRelModel`.

End-to-end `fit` with `build_dfs_features` and the TFM stubbed, so the knobs'
effect on the fitted estimator (extra feature columns, context size, the recency-pool
SUBSAMPLE_SAMPLES) is observable without a real TabPFN or a DFS engine.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table, TaskType

from relarena.cache import CacheConfig
from relarena.models._shared.tfm import tfm
from relarena.models.tabpfn_rel import model as model_mod
from relarena.models.tabpfn_rel.context import hard_pool_subsample_indices
from relarena.models.tabpfn_rel.model import (
    TABPFN_REL_CLIENT_SPACE,
    TABPFN_REL_LOCAL_SPACE,
    TabPFNRelClientModel,
    TabPFNRelLocalModel,
    TabPFNRelModel,
)
from relarena.registry import registry

_N = 100


# -- search space / v3 backend ------------------------------------------------


def test__local_space__default_is_the_validated_config() -> None:
    assert registry.get("tabpfn-rel-local") is TabPFNRelLocalModel
    assert registry.search_space("tabpfn-rel-local") is TABPFN_REL_LOCAL_SPACE
    default = TABPFN_REL_LOCAL_SPACE.default_overrides
    assert default == {
        "tfm": "tabpfn-v3",
        "context_strategy": "hard_pool",
        "subsample_samples": 100_000,
        "pool_inflation": 4.0,
        "with_text": False,
        "max_depth": 2,
    }
    grid = TABPFN_REL_LOCAL_SPACE.fixed_grid
    assert grid[0] == default  # default first, survives budget truncation
    assert [c["max_depth"] for c in grid] == [2, 3, 4]
    assert all(
        {k: c[k] for k in default if k != "max_depth"}
        == {k: default[k] for k in default if k != "max_depth"}
        for c in grid
    )  # only the depth varies


def test__client_model__registered_with_client_tfm_and_text() -> None:
    assert registry.get("tabpfn-rel-client") is TabPFNRelClientModel
    assert registry.search_space("tabpfn-rel-client") is TABPFN_REL_CLIENT_SPACE
    default = TABPFN_REL_CLIENT_SPACE.default_overrides
    assert default["tfm"] == "tabpfn-v3-api"
    assert default["with_text"] is True


def test__tfm_registry__has_v3_with_100k_cap() -> None:
    assert "tabpfn-v3" in tfm.TFM_REGISTRY
    assert tfm.TFM_REGISTRY["tabpfn-v3"].max_train_samples == 100_000


# -- config -> feature pipeline ----------------------------------------------


def test__no_knobs__feature_pipeline_disabled() -> None:
    model = TabPFNRelModel({"tfm": "tabpfn-v2"})
    model._features = model_mod.FeaturePipeline(model.config)
    assert not model._features.enabled


def test__knobs__assemble_calendar_history_text() -> None:
    pipe = model_mod.FeaturePipeline(
        {
            "with_calendar_features": True,
            "with_history_features": True,
            "n_lags": 3,
            "with_text": True,
        }
    )
    assert pipe.calendar and pipe.n_lags == 3 and pipe.text is not None


# -- end-to-end fit (DFS + TFM stubbed) --------------------------------------


class _StubClassifier:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_StubClassifier":
        self.classes_ = np.unique(y)
        self.n_train_ = len(X)
        self.cols_ = list(X.columns)
        self.X_ = X.reset_index(drop=True)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = np.arange(1, len(self.classes_) + 1, dtype=float)
        return np.tile(cols / cols.sum(), (len(X), 1))


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, object]]:
    """Stub the TFM (recording ctor kwargs) and DFS (fixed frame); yield the kwargs."""
    captured: dict[str, object] = {}

    def _make(**kw: object) -> _StubClassifier:
        captured.update(kw)
        return _StubClassifier()

    tfm.TFM_REGISTRY["capture"] = tfm.TFMSpec(
        make_classifier=_make,
        make_regressor=_make,
        max_train_samples=10,
        supports_text=True,
    )
    monkeypatch.setattr(model_mod, "build_dfs_features", _dfs_stub)
    try:
        yield captured
    finally:
        del tfm.TFM_REGISTRY["capture"]


def _dfs_stub(
    task: object, db: object, table: Table, **kwargs: object
) -> tuple[pd.DataFrame, list[str]]:
    # One deterministic feature per row of the passed table, so tests can compare
    # which rows the TFM was fit on by value.
    eid = table.df["eid"].to_numpy(dtype=float)
    feat = pd.DataFrame({"num": eid * 2.0, "cat": np.where(eid % 2, "a", "b")})
    return feat, ["cat"]


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_col="label",
        time_col="ts",
        entity_col="eid",
    )


def _train_table() -> Table:
    df = pd.DataFrame(
        {
            "eid": np.arange(_N),
            "ts": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(_N), "D"),
            "label": [0, 1] * (_N // 2),
        }
    )
    return Table(df=df, fkey_col_to_pkey_table={}, pkey_col=None, time_col="ts")


def test__cache_config__is_passed_explicitly_during_fit(
    capture: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model's explicit immutable cache config reaches DFS unchanged."""
    seen: dict[str, CacheConfig] = {}

    def _recording_dfs(
        task: object, db: object, table: Table, **kwargs: object
    ) -> tuple[pd.DataFrame, list[str]]:
        seen["cache"] = kwargs["cache"]  # type: ignore[assignment]
        return _dfs_stub(task, db, table, **kwargs)

    monkeypatch.setattr(model_mod, "build_dfs_features", _recording_dfs)

    TabPFNRelModel({"tfm": "capture"}, cache=CacheConfig(tmp_path, "raise")).fit(
        _task(), db=None, train_table=_train_table(), val_table=None, seed=0
    )
    assert seen["cache"] == CacheConfig(tmp_path, "raise")

    seen.clear()
    TabPFNRelModel({"tfm": "capture"}).fit(
        _task(), db=None, train_table=_train_table(), val_table=None, seed=0
    )
    assert seen["cache"] == CacheConfig(None, "compute")


def test__random_strategy__caps_via_base_fit_tfm(capture: dict[str, object]) -> None:
    model = TabPFNRelModel({"tfm": "capture"})  # default context = random
    model.fit(_task(), db=None, train_table=_train_table(), val_table=None, seed=0)
    assert model._fitted.estimator.n_train_ == 10  # base fit_tfm cap (stub)
    assert "inference_config" not in capture  # no pool overrides


def test__random_strategy__fits_the_rows_fit_tfm_downsamples_to(
    capture: dict[str, object],
) -> None:
    table = _train_table()
    model = TabPFNRelModel({"tfm": "capture"})
    model.fit(_task(), db=None, train_table=table, val_table=None, seed=7)

    full_feat, _ = _dfs_stub(None, None, table)
    idx = tfm._downsample_indices(
        table.df["label"].to_numpy(),
        TaskType.BINARY_CLASSIFICATION,
        10,
        np.random.default_rng(7),
    )
    pd.testing.assert_frame_equal(
        model._fitted.estimator.X_, full_feat.iloc[idx].reset_index(drop=True)
    )


def test__hard_pool__forwards_indices_and_fits_the_context_union(
    capture: dict[str, object],
) -> None:
    table = _train_table()
    model = TabPFNRelModel(
        {
            "tfm": "capture",
            "context_strategy": "hard_pool",
            "subsample_samples": 8,
            "n_estimators": 4,
            "pool_inflation": 4.0,
        }
    )
    model.fit(_task(), db=None, train_table=table, val_table=None, seed=0)

    ss = capture["inference_config"]["SUBSAMPLE_SAMPLES"]  # type: ignore[index]
    assert isinstance(ss, list) and len(ss) == 4 and all(len(a) == 8 for a in ss)
    assert capture["n_estimators"] == 4

    idx = hard_pool_subsample_indices(
        table.df["ts"].to_numpy(), K=8, M=32, n_estimators=4, seed=0
    )
    union = np.unique(np.concatenate(idx))
    assert model._fitted.estimator.n_train_ == len(union)
    # The fitted frame is the union of contexts; each estimator's SUBSAMPLE_SAMPLES
    # decode back to the right original rows.
    full_feat, _ = _dfs_stub(None, None, table)
    X = model._fitted.estimator.X_
    for got, want in zip(ss, idx):
        pd.testing.assert_frame_equal(
            X.iloc[got].reset_index(drop=True),
            full_feat.iloc[want].reset_index(drop=True),
        )


def test__pool_strategy__requires_subsample_samples(capture: dict[str, object]) -> None:
    model = TabPFNRelModel({"tfm": "capture", "context_strategy": "hard_pool"})
    with pytest.raises(ValueError, match="subsample_samples"):
        model.fit(_task(), db=None, train_table=_train_table(), val_table=None, seed=0)


def test__calendar_knob__adds_columns_to_the_fitted_frame(
    capture: dict[str, object],
) -> None:
    model = TabPFNRelModel({"tfm": "capture", "with_calendar_features": True})
    model.fit(_task(), db=None, train_table=_train_table(), val_table=None, seed=0)
    cols = model._fitted.estimator.cols_
    assert "cutoff_month_sin" in cols and "cutoff_dow_cos" in cols


def test__warm_cache__runs_featurization_without_fitting_a_tfm(
    capture: dict[str, object],
) -> None:
    model = TabPFNRelModel({"tfm": "capture", "with_calendar_features": True})
    table = _train_table()
    eval_tbl = Table(
        df=table.df.iloc[:5].reset_index(drop=True),
        fkey_col_to_pkey_table={},
        pkey_col=None,
        time_col="ts",
    )
    model.warm_cache(_task(), db=None, train_table=table, eval_table=eval_tbl)
    assert not hasattr(model, "_fitted")  # featurization only, no TFM fit
    assert model._features.calendar


def test__predict__returns_one_score_per_row(capture: dict[str, object]) -> None:
    model = TabPFNRelModel({"tfm": "capture", "with_calendar_features": True})
    table = _train_table()
    model.fit(_task(), db=None, train_table=table, val_table=None, seed=0)
    pred = model.predict(
        _task(),
        db=None,
        table=Table(
            df=table.df.iloc[:5].reset_index(drop=True),
            fkey_col_to_pkey_table={},
            pkey_col=None,
            time_col="ts",
        ),
    )
    assert pred.shape == (5,)


# -- text knob wiring ----------------------------------------------------------


def _task_with_text() -> SimpleNamespace:
    task = _task()
    task.entity_table = "users"
    return task


def _db_with_text() -> SimpleNamespace:
    users = SimpleNamespace(
        df=pd.DataFrame(
            {"eid": np.arange(_N), "bio": [f"user number {i}" for i in range(_N)]}
        ),
        pkey_col="eid",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    return SimpleNamespace(table_dict={"users": users})


def test__with_text__passes_raw_text_columns_to_the_estimator(
    capture: dict[str, object],
) -> None:
    model = TabPFNRelModel({"tfm": "capture", "with_text": True})
    model.fit(
        _task_with_text(),
        db=_db_with_text(),
        train_table=_train_table(),
        val_table=None,
        seed=0,
    )

    assert "bio__raw_text" in model._fitted.estimator.cols_


def test__fit__with_text_on_a_textless_tfm__raises(
    capture: dict[str, object],
) -> None:
    # Raise rather than silently running without text: the recorded config would
    # otherwise claim with_text for a run that did not use it.
    # The `capture` fixture owns the registry key and deletes it on teardown.
    tfm.TFM_REGISTRY["capture"] = dataclasses.replace(
        tfm.TFM_REGISTRY["capture"], supports_text=False
    )
    model = TabPFNRelModel({"tfm": "capture", "with_text": True})
    with pytest.raises(ValueError, match="does not support text"):
        model.fit(
            _task_with_text(),
            db=_db_with_text(),
            train_table=_train_table(),
            val_table=None,
            seed=0,
        )
