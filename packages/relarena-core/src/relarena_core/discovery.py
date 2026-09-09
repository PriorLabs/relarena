"""Load installed model modules so their decorators populate the shared registry."""

from __future__ import annotations

from importlib.metadata import entry_points
from threading import RLock

_loaded: set[tuple[str, str]] = set()
_lock = RLock()


def discover_models() -> None:
    """Import model modules declared in the relarena.models entry-point group.

    Module imports execute registration decorators against the shared registry.
    Successful entries load once per process; failed imports remain retryable
    and raise an error naming the plugin. Importing core does not run discovery.
    """
    with _lock:
        for entry in sorted(
            entry_points(group="relarena.models"), key=lambda e: (e.name, e.value)
        ):
            key = (entry.name, entry.value)
            if key in _loaded:
                continue
            try:
                entry.load()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not register model plugin {entry.name!r} "
                    f"({entry.value}). Check its installation and dependencies."
                ) from exc
            _loaded.add(key)


__all__ = ["discover_models"]
