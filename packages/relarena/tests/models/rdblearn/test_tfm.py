"""Model-owned TFM recipes and backend construction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from relarena.models.rdblearn import tfm


def test__rdblearn_tfm__import__registers_model_without_loading_backends() -> None:
    code = """
import sys
from relarena.models.rdblearn import RDBLearnModel, tfm
from relarena.core.registry import registry

prefixes = ('tabpfn', 'tabpfn_client', 'fastdfs')
loaded = [name for name in sys.modules
          if any(name == p or name.startswith(p + '.') for p in prefixes)]
assert not loaded, loaded
assert registry.get('rdblearn') is RDBLearnModel
assert callable(tfm.fit_tfm)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tabpfn_v3_spec_has_no_text_support() -> None:
    # RDBLearn supplies typed numeric and categorical features.
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


def test__importing_the_model_registry__does_not_import_tabpfn() -> None:
    # A subprocess observes registration with no backend already imported.

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import relarena.models; "
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
