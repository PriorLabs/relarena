"""Unit tests for the shared sklearn-output -> evaluate-contract reshaping.

`predict_to_contract` is used by both the `constant-global` and `rdblearn` baselines, so
it is tested here against minimal stub estimators (no real model) rather than through
either wrapper. Covers the shape contract and the class-id alignment edge cases.
"""

from __future__ import annotations

import numpy as np
from relbench.base import TaskType

from relarena.models._shared.predict_contract import predict_to_contract


class _StubClassifier:
    """Exposes fixed `classes_` and a deterministic `predict_proba`."""

    def __init__(self, classes: list[int]) -> None:
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # row = [1, 2, ..., k] normalised, so columns are distinct and ordered.
        cols = np.arange(1, len(self.classes_) + 1, dtype=float)
        return np.tile(cols / cols.sum(), (len(X), 1))


class _StubRegressor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 1.5)


def test_regression_returns_flat_vector() -> None:
    pred = predict_to_contract(_StubRegressor(), np.zeros((8, 2)), TaskType.REGRESSION)
    assert pred.shape == (8,)


def test_binary_returns_positive_class_probability() -> None:
    pred = predict_to_contract(
        _StubClassifier([0, 1]), np.zeros((20, 2)), TaskType.BINARY_CLASSIFICATION
    )
    assert pred.shape == (20,)
    # classes [0, 1] -> proba [1/3, 2/3] -> P(y=1) column = 2/3
    assert np.allclose(pred, 2 / 3)


def test_binary_is_zero_when_positive_class_absent_from_training() -> None:
    pred = predict_to_contract(
        _StubClassifier([0]), np.zeros((5, 2)), TaskType.BINARY_CLASSIFICATION
    )
    assert np.allclose(pred, 0.0)  # class 1 never seen -> P(y=1) = 0
