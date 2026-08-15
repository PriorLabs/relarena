"""Tests for the LightGBM baseline (search space + small fit/predict, no download)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from relbench.base import Table, TaskType

from relarena.models.lightgbm import LIGHTGBM_SPACE, LightGBMModel

_SPACE_KEYS = {
    "n_estimators",
    "learning_rate",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "num_leaves",
    "min_data_in_leaf",
    "extra_trees",
    "min_data_per_group",
    "cat_l2",
    "cat_smooth",
    "max_cat_to_onehot",
    "lambda_l1",
    "lambda_l2",
}


def test_default_config_is_empty() -> None:
    assert LIGHTGBM_SPACE.default_overrides == {}


def test_search_space_keys_ranges_and_reproducibility() -> None:
    configs = LIGHTGBM_SPACE.configs(n_trials=5, seed=0)
    assert configs[0] == {}  # the default config comes first
    sampled = configs[1:]
    assert len(sampled) == 5
    for cfg in sampled:
        assert set(cfg) == _SPACE_KEYS
        assert 50 <= cfg["n_estimators"] <= 1000
        assert 0.4 <= cfg["feature_fraction"] <= 1.0
        assert 0.7 <= cfg["bagging_fraction"] <= 1.0
        assert isinstance(cfg["extra_trees"], bool)
    # reproducible: same seed -> same sampled configs
    assert LIGHTGBM_SPACE.configs(n_trials=5, seed=0) == configs


def _binary_setup(n: int = 80, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    uids = np.arange(n)
    age = rng.integers(18, 70, n).astype(float)
    entity = Table(
        df=pd.DataFrame(
            {"uid": uids, "country": rng.choice(["US", "DE", "FR"], n), "age": age}
        ),
        fkey_col_to_pkey_table={},
        pkey_col="uid",
        time_col=None,
    )
    y = (rng.random(n) < np.where(age > 45, 0.8, 0.2)).astype(int)
    label = Table(
        df=pd.DataFrame({"uid": uids, "date": pd.Timestamp("2021-01-01"), "y": y}),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="date",
    )
    db = SimpleNamespace(table_dict={"users": entity})
    task = SimpleNamespace(
        entity_col="uid",
        entity_table="users",
        target_col="y",
        time_col="date",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    return task, db, label


def test_binary_fit_predict_shape_and_range() -> None:
    task, db, label = _binary_setup()
    model = LightGBMModel({"n_estimators": 20})
    model.fit(task, db, label, None, seed=0)
    pred = model.predict(task, db, label)
    assert pred.shape == (len(label.df),)
    assert pred.min() >= 0.0 and pred.max() <= 1.0
