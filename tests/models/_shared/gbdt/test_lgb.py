"""Tests for the shared native-LightGBM estimator helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from relbench.base import TaskType

from relarena.models._shared.gbdt.lgb import fit_lgb, predict_lgb


def test_feature_names_are_positional_for_lightgbm(monkeypatch: Any) -> None:
    """DFS expression names are hidden from LightGBM but retained for alignment."""
    seen: dict[str, Any] = {}

    class FakeDataset:
        def __init__(
            self,
            data: pd.DataFrame,
            *,
            label: np.ndarray,
            categorical_feature: list[str] | str,
        ) -> None:
            seen["train_columns"] = list(data.columns)
            seen["categorical_feature"] = categorical_feature
            seen["label"] = label

    class FakeBooster:
        def predict(self, data: pd.DataFrame) -> np.ndarray:
            seen["predict_columns"] = list(data.columns)
            seen["predict_values"] = data.to_numpy().copy()
            return np.zeros(len(data))

    def fake_train(
        params: dict[str, Any], dataset: FakeDataset, *, num_boost_round: int
    ) -> FakeBooster:
        return FakeBooster()

    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(Dataset=FakeDataset, train=fake_train),
    )

    invalid = 'COUNT(events["kind"]), value'
    train = pd.DataFrame({invalid: ["a", "b"], "plain": [1.0, 2.0]})
    fitted = fit_lgb(
        train,
        [invalid],
        pd.Series([0, 1]),
        TaskType.BINARY_CLASSIFICATION,
        seed=0,
    )

    assert seen["train_columns"] == ["f0", "f1"]
    assert seen["categorical_feature"] == ["f0"]
    assert fitted.feature_cols == [invalid, "plain"]

    # Deliberately reverse the caller's columns: prediction must restore training
    # order before applying the same safe positional names.
    predict_lgb(fitted, pd.DataFrame({"plain": [3.0], invalid: ["a"]}))
    assert seen["predict_columns"] == ["f0", "f1"]
    assert seen["predict_values"].tolist() == [["a", 3.0]]
