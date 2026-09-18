"""Shared estimator fitting, sampling, alignment and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena_core.tfm import TFMSpec, _downsample_indices, fit_tfm, predict_tfm


class _StubClassifier:
    """Minimal sklearn-like classifier: learns classes_ from y, emits fixed proba."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_StubClassifier":
        self.classes_ = np.unique(y)
        self.n_train_ = len(X)
        self.fit_frame_ = X.copy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_frame_ = X.copy()
        cols = np.arange(1, len(self.classes_) + 1, dtype=float)
        return np.tile(cols / cols.sum(), (len(X), 1))


class _StubRegressor:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_StubRegressor":
        self.mean_ = float(np.mean(y))
        self.n_train_ = len(X)
        self.fit_frame_ = X.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_frame_ = X.copy()
        return np.full(len(X), self.mean_)


@pytest.fixture
def stub_tfm() -> TFMSpec:
    spec = TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _StubRegressor(),
        max_train_samples=10000,
    )
    return spec


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"num": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)}
    )


def test_downsample_classification_caps_and_keeps_every_class() -> None:
    y = np.array([0] * 25 + [1] * 25)
    idx = _downsample_indices(
        y, TaskType.BINARY_CLASSIFICATION, 10, np.random.default_rng(0)
    )
    assert len(idx) == 10
    assert set(y[idx]) == {0, 1}


def test_downsample_regression_caps_and_is_seeded() -> None:
    y = np.arange(100.0)
    a = _downsample_indices(y, TaskType.REGRESSION, 20, np.random.default_rng(0))
    b = _downsample_indices(y, TaskType.REGRESSION, 20, np.random.default_rng(0))
    assert len(a) == 20 and np.array_equal(a, b)


def test_fit_predict_roundtrip_reindexes_to_training_columns(stub_tfm: TFMSpec) -> None:
    df = _frame(20)
    y = pd.Series([0, 1] * 10)
    fitted = fit_tfm(df, y, TaskType.BINARY_CLASSIFICATION, spec=stub_tfm, seed=0)
    pred = predict_tfm(fitted, df[["cat", "num"]])
    assert pred.shape == (20,)


def test_fit_does_not_pass_categorical_indices() -> None:
    captured: dict[str, object] = {}

    def _make(**kw: object) -> _StubClassifier:
        captured.update(kw)
        return _StubClassifier()

    spec = TFMSpec(make_classifier=_make, make_regressor=_make, max_train_samples=10000)
    df = _frame(20)
    y = pd.Series([0, 1] * 10)
    fit_tfm(df, y, TaskType.BINARY_CLASSIFICATION, spec=spec, seed=0)
    assert "categorical_features_indices" not in captured


def test_predict_regression_requests_median_when_supported() -> None:
    captured: dict[str, object] = {}

    class _OutputTypeAwareRegressor(_StubRegressor):
        def predict(self, X: pd.DataFrame, **kwargs: object) -> np.ndarray:
            captured["output_type"] = kwargs.get("output_type")
            return np.full(len(X), self.mean_)

    spec = TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _OutputTypeAwareRegressor(),
        max_train_samples=10000,
    )
    df = _frame(20)
    y = pd.Series(np.arange(20.0))
    fitted = fit_tfm(df, y, TaskType.REGRESSION, spec=spec, seed=0)
    pred = predict_tfm(fitted, df)
    assert captured["output_type"] == "median"
    assert pred.shape == (20,)


def test_predict_regression_plain_estimator_without_output_type(
    stub_tfm: TFMSpec,
) -> None:
    df = _frame(20)
    y = pd.Series(np.arange(20.0))
    fitted = fit_tfm(df, y, TaskType.REGRESSION, spec=stub_tfm, seed=0)
    pred = predict_tfm(fitted, df)
    assert pred.shape == (20,)


def test_predict_uses_the_callers_prediction_batch_limit() -> None:
    batch_lengths: list[int] = []

    class _BatchRecordingRegressor(_StubRegressor):
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            batch_lengths.append(len(X))
            return super().predict(X)

    spec = TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _BatchRecordingRegressor(),
        max_train_samples=10000,
    )
    df = _frame(8)
    fitted = fit_tfm(
        df,
        pd.Series(np.arange(8.0)),
        TaskType.REGRESSION,
        spec=spec,
        seed=0,
        max_predict_samples=3,
    )
    pred = predict_tfm(fitted, df)
    assert batch_lengths == [3, 3, 2]
    assert pred.shape == (8,)


def test_fit_uses_the_tfms_own_sample_cap() -> None:
    spec = TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _StubRegressor(),
        max_train_samples=5,
    )
    df = _frame(50)
    y = pd.Series([0] * 25 + [1] * 25)
    fitted = fit_tfm(df, y, TaskType.BINARY_CLASSIFICATION, spec=spec, seed=0)
    assert fitted.estimator.n_train_ == 5


@pytest.mark.parametrize(
    "task_type", [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]
)
def test_all_missing_columns_are_removed_from_fit_and_predict(
    stub_tfm: TFMSpec, task_type: TaskType
) -> None:
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "sparse": pd.Series([np.nan] * 4, dtype=object),
            "nullable": pd.Series([pd.NA] * 4, dtype="string"),
            "partial": [np.nan, 1.0, np.nan, 2.0],
            "constant": [7.0] * 4,
        }
    )
    original = train.copy(deep=True)
    fitted = fit_tfm(train, pd.Series([0, 1, 0, 1]), task_type, spec=stub_tfm, seed=0)
    expected = train[["num", "partial", "constant"]]
    assert fitted.feature_cols == list(expected.columns)
    pd.testing.assert_frame_equal(fitted.estimator.fit_frame_, expected)
    prediction = train.assign(sparse="general", nullable="business")
    predict_tfm(fitted, prediction[prediction.columns[::-1]])
    pd.testing.assert_frame_equal(fitted.estimator.predict_frame_, expected)
    pd.testing.assert_frame_equal(train, original)
    assert prediction["sparse"].eq("general").all()


def test_all_missing_columns_are_detected_after_sampling(stub_tfm: TFMSpec) -> None:
    y = pd.Series(np.arange(20.0))
    idx = _downsample_indices(
        y.to_numpy(), TaskType.REGRESSION, 5, np.random.default_rng(0)
    )
    train = pd.DataFrame({"num": np.arange(20.0), "sparse": ["general"] * 20})
    train.loc[idx, "sparse"] = np.nan
    assert train["sparse"].notna().any()
    fitted = fit_tfm(
        train, y, TaskType.REGRESSION, spec=stub_tfm, seed=0, max_train_samples=5
    )
    assert fitted.feature_cols == ["num"]
    pd.testing.assert_frame_equal(fitted.estimator.fit_frame_, train.iloc[idx][["num"]])
    predict_tfm(fitted, train)
    pd.testing.assert_frame_equal(fitted.estimator.predict_frame_, train[["num"]])


def test_all_missing_training_frame_raises(stub_tfm: TFMSpec) -> None:
    with pytest.raises(ValueError, match="No features remain"):
        fit_tfm(
            pd.DataFrame({"empty": [np.nan] * 4}),
            pd.Series([0, 1, 0, 1]),
            TaskType.BINARY_CLASSIFICATION,
            spec=stub_tfm,
            seed=0,
        )
