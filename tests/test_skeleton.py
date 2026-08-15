"""Skeleton smoke tests — no data download required.

These exercise the pure-Python contract (metrics, registry, result schema, the
abstract base class). End-to-end tests that load a RelBench task and run the
tuner come with the first concrete model.
"""

from __future__ import annotations

import numpy as np
import pytest

from relarena.metrics import get_metric, is_better, is_higher_better
from relarena.model import RelArenaModel
from relarena.registry import ModelRegistry
from relarena.results import TrialResult, config_id_for, trials_to_dataframe
from relarena.search_space import SearchSpace


def test_metric_direction() -> None:
    assert is_higher_better("roc_auc") is True
    assert is_higher_better("mae") is False
    assert get_metric("r2").higher_is_better is True
    with pytest.raises(KeyError):
        is_higher_better("not_a_metric")


def test_is_better_respects_direction() -> None:
    assert is_better(0.9, 0.8, "roc_auc") is True  # higher better
    assert is_better(0.1, 0.2, "mae") is True  # lower better


def test_config_id_is_order_independent_and_distinct() -> None:
    assert config_id_for({"x": 1, "y": 2}) == config_id_for({"y": 2, "x": 1})
    assert config_id_for({"x": 1}) != config_id_for({"x": 2})


def test_registry_register_get_iter() -> None:
    reg = ModelRegistry()

    class M(RelArenaModel):
        name = "m"

        def fit(self, *a, **k) -> None:  # noqa: D401
            ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    reg.register(M, SearchSpace(default_overrides={}))
    assert reg.get("m") is M
    assert "m" in reg
    assert reg.names() == ["m"]
    assert list(reg) == [M]
    assert len(reg) == 1


def test_registry_rejects_duplicate_name() -> None:
    reg = ModelRegistry()

    class A(RelArenaModel):
        name = "dup"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    class B(RelArenaModel):
        name = "dup"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    reg.register(A, SearchSpace(default_overrides={}))
    with pytest.raises(ValueError):
        reg.register(B, SearchSpace(default_overrides={}))


def test_abstract_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        RelArenaModel()


def test__refit_on_full_data__defaults_true_and_is_overridable() -> None:
    class Default(RelArenaModel):
        name = "default-regime"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    class BestVal(Default):
        name = "best-val-regime"
        refit_on_full_data = False

    assert RelArenaModel.refit_on_full_data is True  # harness default
    assert Default.refit_on_full_data is True
    assert BestVal.refit_on_full_data is False


def test_trials_to_dataframe_drops_arrays() -> None:
    t = TrialResult(
        config={"a": 1},
        config_id=config_id_for({"a": 1}),
        config_tag="default",
        val_score=0.5,
        val_pred=np.zeros(3),
    )
    df = trials_to_dataframe([t])
    assert "val_pred" not in df.columns
    assert df.loc[0, "val_score"] == 0.5


def test_trials_to_dataframe_flattens_metrics() -> None:
    t = TrialResult(
        config={},
        config_id=config_id_for({}),
        config_tag="default",
        val_score=0.5,
        test_score=0.6,
        val_metrics={"mae": 0.5, "rmse": 0.7},
        test_metrics={"mae": 0.6, "rmse": 0.8},
    )
    df = trials_to_dataframe([t])
    # the metric dicts themselves are not columns; their entries are flattened.
    assert "val_metrics" not in df.columns and "test_metrics" not in df.columns
    assert df.loc[0, "val_mae"] == 0.5 and df.loc[0, "val_rmse"] == 0.7
    assert df.loc[0, "test_mae"] == 0.6 and df.loc[0, "test_rmse"] == 0.8
