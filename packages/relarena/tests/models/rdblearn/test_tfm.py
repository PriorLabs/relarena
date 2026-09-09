"""RDBLearn TFM recipes and local backend construction."""

from __future__ import annotations

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relarena.models.rdblearn import tfm


def test__rdblearn_tfm__import__registers_model_without_loading_backends() -> None:
    code = """
import sys
from relarena.models.rdblearn import RDBLearnModel, tfm
from relarena_core.registry import registry

prefixes = ('tabpfn_rel', 'tabpfn', 'tabpfn_client', 'fastdfs')
loaded = [name for name in sys.modules
          if any(name == p or name.startswith(p + '.') for p in prefixes)]
assert not loaded, loaded
assert registry.get('rdblearn') is RDBLearnModel
assert set(tfm.TFM_REGISTRY) == {'tabpfn-v2', 'tabpfn-v2.5'}
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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
