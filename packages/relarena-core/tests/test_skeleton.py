"""Shared model, registry and result contracts."""

from __future__ import annotations

import numpy as np
import pytest

from relarena_core.metrics import get_metric, is_better, is_higher_better
from relarena_core.model import RelArenaModel
from relarena_core.registry import ModelRegistry
from relarena_core.results import config_id_for
from relarena_core.search_space import SearchSpace
from relarena_core.system import RelArenaSystem


def test_metric_direction() -> None:
    assert is_higher_better("roc_auc") is True
    assert is_higher_better("mae") is False
    assert get_metric("r2").higher_is_better is True
    with pytest.raises(KeyError):
        is_higher_better("not_a_metric")


def test_is_better_respects_direction() -> None:
    assert is_better(0.9, 0.8, "roc_auc") is True  # higher better
    assert is_better(0.1, 0.2, "mae") is True  # lower better


def test_config_id_is_order_independent_and_distinct() -> None:
    assert config_id_for({"x": 1, "y": 2}) == config_id_for({"y": 2, "x": 1})
    assert config_id_for({"x": 1}) != config_id_for({"x": 2})


def test_registry_register_get_iter() -> None:
    reg = ModelRegistry()

    class M(RelArenaModel):
        name = "m"

        def fit(self, *a, **k) -> None:  # noqa: D401
            ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    reg.register(M, SearchSpace(default_overrides={}))
    assert reg.get("m") is M
    assert "m" in reg
    assert reg.names() == ["m"]
    assert list(reg) == [M]
    assert len(reg) == 1


def test_registry_rejects_duplicate_name() -> None:
    reg = ModelRegistry()

    class A(RelArenaModel):
        name = "dup"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    class B(RelArenaModel):
        name = "dup"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    reg.register(A, SearchSpace(default_overrides={}))
    with pytest.raises(ValueError):
        reg.register(B, SearchSpace(default_overrides={}))


def test_registry_registers_system_without_search_space() -> None:
    reg = ModelRegistry()

    class S(RelArenaSystem):
        name = "s"

        def run(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    reg.register_system(S)

    assert reg.get("s") is S
    assert reg.kind("s") == "system"
    with pytest.raises(TypeError, match="no harness search space"):
        reg.search_space("s")


def _dated_models() -> tuple[type[RelArenaModel], type[RelArenaModel]]:
    class Old(RelArenaModel):
        name = "m-2026-01-01"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    class New(Old):
        name = "m-2026-02-01"

    return Old, New


def test_registry_alias_resolves_to_the_dated_method() -> None:
    reg = ModelRegistry()
    old, new = _dated_models()
    space = SearchSpace(default_overrides={"x": 1})
    reg.register(old, SearchSpace(default_overrides={}))
    reg.register(new, space)

    reg.register_alias("m-latest", new.name)

    assert reg.get("m-latest") is new
    assert reg.get("m-latest").name == "m-2026-02-01"
    assert reg.search_space("m-latest") is space
    assert reg.kind("m-latest") == "model"
    assert reg.resolve("m-latest") == new.name
    assert reg.resolve(old.name) == old.name
    assert reg.alias_for(new.name) == "m-latest"
    assert reg.alias_for(old.name) is None
    assert "m-latest" in reg
    assert reg.names() == [old.name, new.name]


def test_registry_alias_rejects_conflicts() -> None:
    reg = ModelRegistry()
    old, new = _dated_models()
    reg.register(old, SearchSpace(default_overrides={}))
    reg.register(new, SearchSpace(default_overrides={}))
    reg.register_alias("m-latest", new.name)
    reg.register_alias("m-latest", new.name)

    with pytest.raises(KeyError, match="unregistered"):
        reg.register_alias("x-latest", "unknown")
    with pytest.raises(ValueError, match="method name"):
        reg.register_alias(old.name, new.name)
    with pytest.raises(ValueError, match="already points"):
        reg.register_alias("m-latest", old.name)
    with pytest.raises(ValueError, match="already has the alias"):
        reg.register_alias("m-newest", new.name)

    class Clash(RelArenaModel):
        name = "m-latest"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    with pytest.raises(ValueError, match="already an alias"):
        reg.register(Clash, SearchSpace(default_overrides={}))


def test_abstract_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        RelArenaModel()


def test_abstract_system_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        RelArenaSystem()


def test__refit_on_full_data__defaults_true_and_is_overridable() -> None:
    class Default(RelArenaModel):
        name = "default-regime"

        def fit(self, *a, **k) -> None: ...

        def predict(self, *a, **k) -> np.ndarray:
            return np.zeros(1)

    class BestVal(Default):
        name = "best-val-regime"
        refit_on_full_data = False

    assert RelArenaModel.refit_on_full_data is True  # harness default
    assert Default.refit_on_full_data is True
    assert BestVal.refit_on_full_data is False
