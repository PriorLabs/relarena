from importlib.metadata import EntryPoint

import pytest

from relarena import discover_models, discovery, registry


def test_installed_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = EntryPoint(
        name="constant", value="relarena.models.dummy", group="relarena.models"
    )
    monkeypatch.setattr(discovery, "entry_points", lambda **kwargs: [entry])
    discover_models()
    model = registry.get("constant-global")
    discover_models()
    assert registry.get("constant-global") is model


def test_plugin_failure_identifies_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = EntryPoint(
        name="broken", value="missing_model_package", group="relarena.models"
    )
    monkeypatch.setattr(discovery, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(RuntimeError, match="broken") as error:
        discover_models()
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
