"""Installed model modules populate the registry through decorators."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relarena_core import discovery
from relarena_core.model import RelArenaModel
from relarena_core.registry import MethodRegistry, register_model
from relarena_core.search_space import SearchSpace

registry_module = import_module("relarena_core.registry")


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> MethodRegistry:
    registry = MethodRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    monkeypatch.setattr(discovery, "_loaded", set())
    return registry


def _entry(name: str, load: Mock) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=f"{name}.model", load=load)


def test_successful_module_import_runs_once(
    isolated: MethodRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def import_model() -> None:
        register_model(search_space=SearchSpace(default_overrides={}))(
            type("Example", (RelArenaModel,), {"name": "example"})
        )

    entry = _entry("example", Mock(side_effect=import_model))
    entries = Mock(return_value=[entry])
    monkeypatch.setattr(discovery, "entry_points", entries)
    discovery.discover_models()
    discovery.discover_models()
    assert isolated.names() == ["example"]
    entry.load.assert_called_once_with()
    entries.assert_called_with(group="relarena.models")


def test_failed_plugin_is_visible_and_retryable(
    isolated: MethodRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = ModuleNotFoundError("plugin backend is absent", name="example_backend")
    entry = _entry("broken", Mock(side_effect=[missing, None]))
    monkeypatch.setattr(discovery, "entry_points", Mock(return_value=[entry]))
    with pytest.raises(RuntimeError, match="broken.*broken.model") as error:
        discovery.discover_models()
    assert error.value.__cause__ is missing
    discovery.discover_models()
    discovery.discover_models()
    assert entry.load.call_count == 2


def test_duplicate_model_name_is_not_silently_replaced(
    isolated: MethodRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = type("First", (RelArenaModel,), {"name": "same"})
    second = type("Second", (RelArenaModel,), {"name": "same"})
    space = SearchSpace(default_overrides={})
    isolated.register(first, space)
    entry = _entry(
        "collision",
        Mock(side_effect=lambda: register_model(search_space=space)(second)),
    )
    monkeypatch.setattr(discovery, "entry_points", Mock(return_value=[entry]))
    with pytest.raises(RuntimeError, match="collision") as error:
        discovery.discover_models()
    assert isinstance(error.value.__cause__, ValueError)
    assert isolated.get("same") is first


def test_missing_model_explains_discovery() -> None:
    with pytest.raises(KeyError) as error:
        MethodRegistry().get("external-model")
    assert "discover_models()" in str(error.value)
