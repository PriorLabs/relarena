"""Model and system registry.

A string-keyed registry for both method contracts. Model entries pair a class
with its external `SearchSpace`; system entries need no search space because
they own their complete prediction procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Type, TypeAlias

from relarena.model import RelArenaModel
from relarena.search_space import SearchSpaceProvider
from relarena.system import RelArenaSystem

Method: TypeAlias = type[RelArenaModel] | type[RelArenaSystem]


@dataclass(frozen=True)
class RegistryEntry:
    """One registered method and the harness metadata needed to run it."""

    method_cls: Method
    kind: str
    search_space: SearchSpaceProvider | None = None


class MethodRegistry:
    """A string-keyed collection of model and system entries.

    The "search space" of an entry is a `SearchSpaceProvider` — a fixed
    `SearchSpace` or a factory the harness resolves per task (see
    `search_space.py`); the registry stores it opaquely either way.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._entries: dict[str, RegistryEntry] = {}

    def register(
        self, model_cls: Type[RelArenaModel], search_space: SearchSpaceProvider
    ) -> Type[RelArenaModel]:
        """Register `model_cls` (under its class-level `name`) with its space.

        Returns the class unchanged. Raises if `name` is missing or already taken
        by a different class.
        """
        name = getattr(model_cls, "name", None)
        if not name:
            raise ValueError(f"{model_cls.__name__} must define a class-level `name`.")
        existing = self._entries.get(name)
        if existing is not None and existing.method_cls is not model_cls:
            raise ValueError(
                f"A different model is already registered under "
                f"'{name}': {existing.method_cls.__name__}"
            )
        # `kind` preserves compatibility with the experimental pre-system API.
        # Native systems use `register_system` below.
        kind = getattr(model_cls, "kind", "model")
        self._entries[name] = RegistryEntry(model_cls, kind, search_space)
        return model_cls

    def register_system(self, system_cls: Type[RelArenaSystem]) -> Type[RelArenaSystem]:
        """Register a native system, which has no harness search space."""
        name = getattr(system_cls, "name", None)
        if not name:
            raise ValueError(f"{system_cls.__name__} must define a class-level `name`.")
        existing = self._entries.get(name)
        if existing is not None and existing.method_cls is not system_cls:
            raise ValueError(
                f"A different method is already registered under "
                f"'{name}': {existing.method_cls.__name__}"
            )
        self._entries[name] = RegistryEntry(system_cls, "system")
        return system_cls

    def get(self, name: str) -> Method:
        """Return the model or system class under `name` (raises if unknown)."""
        return self._entry(name).method_cls

    def search_space(self, name: str) -> SearchSpaceProvider:
        """Return the search-space provider under `name` (raises if unknown)."""
        entry = self._entry(name)
        if entry.search_space is None:
            raise TypeError(f"System '{name}' has no harness search space.")
        return entry.search_space

    def search_space_for(self, model_cls: Type[RelArenaModel]) -> SearchSpaceProvider:
        """Return the search-space provider for `model_cls` (by its `name`)."""
        return self.search_space(model_cls.name)

    def kind(self, name: str) -> str:
        """Return `"model"` or `"system"` for the registered method."""
        return self._entry(name).kind

    def names(self) -> list[str]:
        """Return the registered method names, sorted."""
        return sorted(self._entries)

    def _entry(self, name: str) -> RegistryEntry:
        if name not in self._entries:
            raise KeyError(
                f"No method registered under '{name}'. Known: {self.names()}"
            )
        return self._entries[name]

    def __iter__(self) -> Iterator[Method]:
        """Iterate over the registered model and system classes."""
        return (entry.method_cls for entry in self._entries.values())

    def __contains__(self, name: object) -> bool:
        """Return whether a method is registered under `name`."""
        return name in self._entries

    def __len__(self) -> int:
        """Return the number of registered methods."""
        return len(self._entries)


#: Backward-compatible name for the registry class.
ModelRegistry = MethodRegistry

#: The default global registry used by both registration decorators.
registry = MethodRegistry()


def register_model(
    *, search_space: SearchSpaceProvider
) -> Callable[[Type[RelArenaModel]], Type[RelArenaModel]]:
    """Decorator binding a model class to its `search_space` in `registry`.

    `search_space` is a fixed `SearchSpace` or a factory
    (`SearchSpaceProvider`) the harness resolves per task. Usage:

        @register_model(search_space=MY_SPACE)
        class MyModel(RelArenaModel): ...
    """

    def decorator(model_cls: Type[RelArenaModel]) -> Type[RelArenaModel]:
        return registry.register(model_cls, search_space)

    return decorator


def register_system(
    system_cls: Type[RelArenaSystem],
) -> Type[RelArenaSystem]:
    """Register a `RelArenaSystem` without a harness search space.

    Usage::

        @register_system
        class MySystem(RelArenaSystem): ...
    """
    return registry.register_system(system_cls)


__all__ = [
    "Method",
    "MethodRegistry",
    "ModelRegistry",
    "RegistryEntry",
    "register_model",
    "register_system",
    "registry",
]
