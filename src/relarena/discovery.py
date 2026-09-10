"""Discovery of built-in and installed model packages."""

from importlib import import_module
from importlib.metadata import entry_points


def discover_models() -> None:
    """Import model modules so their registration decorators run."""
    import_module("relarena.models")
    for entry in entry_points(group="relarena.models"):
        try:
            entry.load()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model entry point {entry.name!r}."
            ) from exc
