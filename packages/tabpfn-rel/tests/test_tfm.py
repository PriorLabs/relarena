"""TabPFN-Rel TFM recipes and hosted backend construction."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import numpy as np
import pytest

from tabpfn_rel import tfm


def test_tabpfn_v3_spec_has_no_text_support() -> None:
    # The local backend receives typed numeric and categorical features only.
    assert not tfm.TFM_REGISTRY["tabpfn-v3"].supports_text


def test_tabpfn_v3_api_spec_builds_the_client_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

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
    assert captured["model_path"] == "v3_default"
    # The API handles raw text natively.
    assert spec.supports_text


def test__make_tabpfn_api__ndarray_subsample_indices__converted_to_int_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

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
