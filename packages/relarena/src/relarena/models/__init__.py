"""Discover baseline modules whose decorators register models and systems.

Model implementations and their search spaces live together in each package.
Importing this package loads each public child package; shared helpers are
excluded. Optional backends load during model execution.
"""

import importlib
import pkgutil


def discover_models() -> None:
    """Import baseline packages and propagate failures from their implementations."""
    for _finder, name, is_package in pkgutil.iter_modules(__path__):
        if is_package and not name.startswith("_"):
            importlib.import_module(f"{__name__}.{name}")


discover_models()
