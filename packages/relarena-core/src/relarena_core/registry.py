"""Model and system registry.

A string-keyed registry for both method contracts. Model entries pair a class
with its external `SearchSpace`; system entries need no search space because
they own their complete prediction procedure. An alias such as
`tabpfn-rel-client-latest` names a registered method; lookups accept it, while
results keep the method's own name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Type, TypeAlias

from relarena_core.model import RelArenaModel
from relarena_core.search_space import SearchSpaceProvider
from relarena_core.system import RelArenaSystem

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
        self._aliases: dict[str, str] = {}

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
        self._check_not_alias(name)
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
        self._check_not_alias(name)
        existing = self._entries.get(name)
        if existing is not None and existing.method_cls is not system_cls:
            raise ValueError(
                f"A different method is already registered under "
                f"'{name}': {existing.method_cls.__name__}"
            )
        self._entries[name] = RegistryEntry(system_cls, "system")
        return system_cls

    def register_alias(self, alias: str, name: str) -> None:
        """Make `alias` resolve to the registered method `name`.

        A method has at most one alias, which leaderboards show as its label.
        Raises if `name` is not registered, if `alias` is a method name, or if
        either already has a different pairing.
        """
        if name not in self._entries:
            raise KeyError(f"Cannot alias unregistered method '{name}'.")
        if alias in self._entries:
            raise ValueError(f"'{alias}' is a method name and cannot be an alias.")
        existing = self._aliases.get(alias)
        if existing is not None and existing != name:
            raise ValueError(f"Alias '{alias}' already points to '{existing}'.")
        current = self.alias_for(name)
        if current is not None and current != alias:
            raise ValueError(f"'{name}' already has the alias '{current}'.")
        self._aliases[alias] = name

    def resolve(self, name: str) -> str:
        """Return the method name that `name` refers to, following an alias."""
        return self._aliases.get(name, name)

    def alias_for(self, name: str) -> str | None:
        """Return the alias pointing to method `name`, or None if it has none."""
        return next((a for a, n in self._aliases.items() if n == name), None)

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

    def _check_not_alias(self, name: str) -> None:
        if name in self._aliases:
            raise ValueError(
                f"'{name}' is already an alias of '{self._aliases[name]}'."
            )

    def _entry(self, name: str) -> RegistryEntry:
        name = self.resolve(name)
        if name not in self._entries:
            hint = (
                " Call relarena_core.discover_models() to load installed model plugins."
            )
            raise KeyError(
                f"No method registered under '{name}'. Known: {self.names()}.{hint}"
            )
        return self._entries[name]

    def __iter__(self) -> Iterator[Method]:
        """Iterate over the registered model and system classes."""
        return (entry.method_cls for entry in self._entries.values())

    def __contains__(self, name: object) -> bool:
        """Return whether `name` is a registered method or an alias of one."""
        return name in self._entries or name in self._aliases

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
