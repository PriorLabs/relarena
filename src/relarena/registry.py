"""Model registry.

A string-keyed registry pairing each model class with its hyperparameter
`SearchSpace` (the search space lives *beside* the
model, not on it — see `search_space.py`). Mirrors TabArena, where a model is
bound to its config generator externally rather than declaring its own.
"""

from __future__ import annotations

from typing import Callable, Iterator, Type

from relarena.model import RelArenaModel
from relarena.search_space import SearchSpaceProvider


class ModelRegistry:
    """A string-keyed collection of `(model class, search space)` entries.

    The "search space" of an entry is a `SearchSpaceProvider` — a fixed
    `SearchSpace` or a factory the harness resolves per task (see
    `search_space.py`); the registry stores it opaquely either way.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._entries: dict[str, tuple[Type[RelArenaModel], SearchSpaceProvider]] = {}

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
        if existing is not None and existing[0] is not model_cls:
            raise ValueError(
                f"A different model is already registered under "
                f"'{name}': {existing[0].__name__}"
            )
        self._entries[name] = (model_cls, search_space)
        return model_cls

    def get(self, name: str) -> Type[RelArenaModel]:
        """Return the model class registered under `name` (raises if unknown)."""
        return self._entry(name)[0]

    def search_space(self, name: str) -> SearchSpaceProvider:
        """Return the search-space provider under `name` (raises if unknown)."""
        return self._entry(name)[1]

    def search_space_for(self, model_cls: Type[RelArenaModel]) -> SearchSpaceProvider:
        """Return the search-space provider for `model_cls` (by its `name`)."""
        return self.search_space(model_cls.name)

    def names(self) -> list[str]:
        """Return the registered model names, sorted."""
        return sorted(self._entries)

    def _entry(self, name: str) -> tuple[Type[RelArenaModel], SearchSpaceProvider]:
        if name not in self._entries:
            raise KeyError(f"No model registered under '{name}'. Known: {self.names()}")
        return self._entries[name]

    def __iter__(self) -> Iterator[Type[RelArenaModel]]:
        """Iterate over the registered model classes."""
        return (cls for cls, _ in self._entries.values())

    def __contains__(self, name: object) -> bool:
        """Return whether a model is registered under `name`."""
        return name in self._entries

    def __len__(self) -> int:
        """Return the number of registered models."""
        return len(self._entries)


#: The default global registry. Models register into it via `@register_model`.
registry = ModelRegistry()


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
