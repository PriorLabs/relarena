"""Unit tests for the TFM estimator core (via a stub TFM, no real TabPFN inference).

A stub estimator is registered into TFM_REGISTRY so the downsample -> fit -> predict
path and the downsampler can be tested without running real TabPFN.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models._shared.tfm import tfm
from relarena.models._shared.tfm.tfm import (
    _downsample_indices,
    fit_tfm,
    predict_tfm,
)


class _StubClassifier:
    """Minimal sklearn-like classifier: learns classes_ from y, emits fixed proba."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_StubClassifier":
        self.classes_ = np.unique(y)
        self.n_train_ = len(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = np.arange(1, len(self.classes_) + 1, dtype=float)
        return np.tile(cols / cols.sum(), (len(X), 1))


class _StubRegressor:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_StubRegressor":
        self.mean_ = float(np.mean(y))
        self.n_train_ = len(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


@pytest.fixture
def stub_tfm() -> Iterator[str]:
    """Register a 'stub' TFM for the duration of a test, then remove it."""
    tfm.TFM_REGISTRY["stub"] = tfm.TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _StubRegressor(),
        max_train_samples=10_000,
    )
    yield "stub"
    del tfm.TFM_REGISTRY["stub"]


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"num": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)}
    )


# -- downsampling ------------------------------------------------------------


def test_downsample_classification_caps_and_keeps_every_class() -> None:
    y = np.array([0] * 25 + [1] * 25)
    idx = _downsample_indices(
        y, TaskType.BINARY_CLASSIFICATION, 10, np.random.default_rng(0)
    )
    assert len(idx) == 10
    assert set(y[idx]) == {0, 1}  # both classes survive


def test_downsample_regression_caps_and_is_seeded() -> None:
    y = np.arange(100.0)
    a = _downsample_indices(y, TaskType.REGRESSION, 20, np.random.default_rng(0))
    b = _downsample_indices(y, TaskType.REGRESSION, 20, np.random.default_rng(0))
    assert len(a) == 20 and np.array_equal(a, b)


# -- fit / predict ------------------------------------------------------------


def test_fit_predict_roundtrip_reindexes_to_training_columns(stub_tfm: str) -> None:
    # predict_tfm reindexes to the training column order, then delegates the output
    # reshaping to predict_to_contract (covered in test_predict_contract.py). Here we
    # only check the fit -> predict wiring survives reordered predict-time columns.
    df = _frame(20)
    y = pd.Series([0, 1] * 10)
    fitted = fit_tfm(df, y, TaskType.BINARY_CLASSIFICATION, tfm=stub_tfm, seed=0)
    pred = predict_tfm(fitted, df[["cat", "num"]])  # columns reordered on purpose
    assert pred.shape == (20,)


def test_fit_does_not_pass_categorical_indices() -> None:
    # The TFM auto-detects categoricals from the DataFrame; fit_tfm must not pre-flag
    # them via categorical_features_indices.
    captured: dict[str, object] = {}

    def _make(**kw: object) -> _StubClassifier:
        captured.update(kw)
        return _StubClassifier()

    tfm.TFM_REGISTRY["capture"] = tfm.TFMSpec(
        make_classifier=_make,
        make_regressor=_make,
        max_train_samples=10_000,
    )
    try:
        df = _frame(20)  # columns: ["num", "cat"]
        y = pd.Series([0, 1] * 10)
        fit_tfm(df, y, TaskType.BINARY_CLASSIFICATION, tfm="capture", seed=0)
        assert "categorical_features_indices" not in captured
    finally:
        del tfm.TFM_REGISTRY["capture"]


def test_predict_regression_requests_median_when_supported() -> None:
    # MAE is the primary regression metric; predict_tfm must request the
    # MAE-optimal median from estimators that support output_type.
    captured: dict[str, object] = {}

    class _OutputTypeAwareRegressor(_StubRegressor):
        def predict(self, X: pd.DataFrame, **kwargs: object) -> np.ndarray:
            captured["output_type"] = kwargs.get("output_type")
            return np.full(len(X), self.mean_)

    tfm.TFM_REGISTRY["median"] = tfm.TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _OutputTypeAwareRegressor(),
        max_train_samples=10_000,
    )
    try:
        df = _frame(20)
        y = pd.Series(np.arange(20.0))
        fitted = fit_tfm(df, y, TaskType.REGRESSION, tfm="median", seed=0)
        pred = predict_tfm(fitted, df)
        assert captured["output_type"] == "median"
        assert pred.shape == (20,)
    finally:
        del tfm.TFM_REGISTRY["median"]


def test_predict_regression_plain_estimator_without_output_type(
    stub_tfm: str,
) -> None:
    # Estimators without an output_type parameter (e.g. a future LimiX wrapper)
    # fall back to the plain predict path.
    df = _frame(20)
    y = pd.Series(np.arange(20.0))
    fitted = fit_tfm(df, y, TaskType.REGRESSION, tfm=stub_tfm, seed=0)
    pred = predict_tfm(fitted, df)
    assert pred.shape == (20,)


def test_predict_uses_the_callers_prediction_batch_limit() -> None:
    batch_lengths: list[int] = []

    class _BatchRecordingRegressor(_StubRegressor):
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            batch_lengths.append(len(X))
            return super().predict(X)

    tfm.TFM_REGISTRY["batched"] = tfm.TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _BatchRecordingRegressor(),
        max_train_samples=10_000,
    )
    try:
        df = _frame(8)
        fitted = fit_tfm(
            df,
            pd.Series(np.arange(8.0)),
            TaskType.REGRESSION,
            tfm="batched",
            seed=0,
            max_predict_samples=3,
        )

        pred = predict_tfm(fitted, df)

        assert batch_lengths == [3, 3, 2]
        assert pred.shape == (8,)
    finally:
        del tfm.TFM_REGISTRY["batched"]


def test_fit_uses_the_tfms_own_sample_cap() -> None:
    # Without an explicit max_train_samples, fit_tfm uses the TFM's registry cap.
    tfm.TFM_REGISTRY["small"] = tfm.TFMSpec(
        make_classifier=lambda **kw: _StubClassifier(),
        make_regressor=lambda **kw: _StubRegressor(),
        max_train_samples=5,
    )
    try:
        df = _frame(50)
        y = pd.Series([0] * 25 + [1] * 25)
        fitted = fit_tfm(df, y, TaskType.BINARY_CLASSIFICATION, tfm="small", seed=0)
        assert fitted.estimator.n_train_ == 5  # spec cap applied
    finally:
        del tfm.TFM_REGISTRY["small"]


def test_tabpfn_v3_spec_has_no_text_support() -> None:
    # The local v3 estimator cannot consume raw text; TabPFNRelModel's with_text
    # guard rests on this staying false.
    assert not tfm.TFM_REGISTRY["tabpfn-v3"].supports_text


def test_tabpfn_v3_api_spec_builds_the_client_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    captured: dict[str, object] = {}

    class _ApiEstimator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("tabpfn_client")
    module.TabPFNClassifier = _ApiEstimator  # type: ignore[attr-defined]
    module.TabPFNRegressor = _ApiEstimator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tabpfn_client", module)

    spec = tfm.TFM_REGISTRY["tabpfn-v3-api"]
    estimator = spec.make_classifier(device="cuda", seed=7)

    assert isinstance(estimator, _ApiEstimator)
    # device is server-side and never forwarded to the client constructor.
    assert captured == {
        "model_path": "v3_default",
        "random_state": 7,
        "ignore_pretraining_limits": True,
    }
    assert isinstance(spec.make_regressor(device="cpu", seed=7), _ApiEstimator)
    # The API handles raw text natively.
    assert spec.supports_text


def test__importing_the_model_registry__does_not_import_tabpfn() -> None:
    # tabpfn ships the Prior Labs License, whose paragraph 10 obliges downstream
    # attribution, so it belongs to the rdblearn extra rather than a core install.
    # Importing the registry must not pull it in even where it is installed, which
    # is what keeps that containment true independently of
    # whichever environment the suite happens to run in.
    #
    # A subprocess, because this test cannot observe a clean import otherwise: the
    # suite has already imported relarena.models, and re-importing it in-process would
    # re-register every model and trip the registry's duplicate-name guard.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import relarena.models, sys; "
            "sys.exit(1 if 'tabpfn' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"importing relarena.models pulled in tabpfn\n{result.stderr}"
    )


def test__local_tabpfn_spec__tabpfn_stubbed__imports_it_only_when_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tabpfn ships in the rdblearn extra, so building the registry must not import it;
    # a stub injected after import still wins, which is what proves the import is lazy.
    import sys
    from types import ModuleType

    captured: dict[str, object] = {}

    class _LocalEstimator:
        @classmethod
        def create_default_for_version(
            cls, version: object, **kwargs: object
        ) -> "_LocalEstimator":
            captured["version"] = version
            captured.update(kwargs)
            return cls()

    module = ModuleType("tabpfn")
    module.TabPFNClassifier = _LocalEstimator  # type: ignore[attr-defined]
    module.TabPFNRegressor = _LocalEstimator  # type: ignore[attr-defined]
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = SimpleNamespace(  # type: ignore[attr-defined]
        V2="v2-checkpoint", V2_5="v2.5-checkpoint", V3="v3-checkpoint"
    )
    settings = ModuleType("tabpfn.settings")
    settings.settings = SimpleNamespace(  # type: ignore[attr-defined]
        tabpfn=SimpleNamespace(max_batched_test_rows=32768)
    )
    monkeypatch.setitem(sys.modules, "tabpfn", module)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    monkeypatch.setitem(sys.modules, "tabpfn.settings", settings)
    monkeypatch.delenv("TABPFN_MAX_BATCHED_TEST_ROWS", raising=False)

    estimator = tfm.TFM_REGISTRY["tabpfn-v2.5"].make_classifier(device="cpu", seed=7)

    assert isinstance(estimator, _LocalEstimator)
    assert "TABPFN_MAX_BATCHED_TEST_ROWS" not in os.environ
    assert settings.settings.tabpfn.max_batched_test_rows == 32768
    assert captured == {
        "version": "v2.5-checkpoint",
        "device": "cpu",
        "random_state": 7,
        "ignore_pretraining_limits": True,
    }


def test__local_tabpfn_spec__does_not_apply_environment_batch_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    class _LocalEstimator:
        @classmethod
        def create_default_for_version(
            cls, version: object, **kwargs: object
        ) -> "_LocalEstimator":
            return cls()

    module = ModuleType("tabpfn")
    module.TabPFNClassifier = _LocalEstimator  # type: ignore[attr-defined]
    module.TabPFNRegressor = _LocalEstimator  # type: ignore[attr-defined]
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = SimpleNamespace(  # type: ignore[attr-defined]
        V2="v2-checkpoint", V2_5="v2.5-checkpoint", V3="v3-checkpoint"
    )
    settings = ModuleType("tabpfn.settings")
    settings.settings = SimpleNamespace(  # type: ignore[attr-defined]
        tabpfn=SimpleNamespace(max_batched_test_rows=32768)
    )
    monkeypatch.setitem(sys.modules, "tabpfn", module)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    monkeypatch.setitem(sys.modules, "tabpfn.settings", settings)
    monkeypatch.setenv("TABPFN_MAX_BATCHED_TEST_ROWS", "4096")

    tfm.TFM_REGISTRY["tabpfn-v2"].make_classifier(device="cpu", seed=0)

    assert os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] == "4096"
    assert settings.settings.tabpfn.max_batched_test_rows == 32768


def test__make_tabpfn_api__ndarray_subsample_indices__converted_to_int_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import sys
    from types import ModuleType

    captured: dict[str, object] = {}

    class _ApiEstimator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = ModuleType("tabpfn_client")
    module.TabPFNClassifier = _ApiEstimator  # type: ignore[attr-defined]
    module.TabPFNRegressor = _ApiEstimator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tabpfn_client", module)

    tfm.TFM_REGISTRY["tabpfn-v3-api"].make_classifier(
        device="cpu",
        seed=0,
        n_estimators=2,
        inference_config={"SUBSAMPLE_SAMPLES": [np.array([0, 2]), np.array([1, 3])]},
    )

    config = captured["inference_config"]
    assert config["SUBSAMPLE_SAMPLES"] == [[0, 2], [1, 3]]
    json.dumps(config)  # what tabpfn_client serializes into the request body
