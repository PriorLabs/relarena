"""Discover baseline modules whose decorators register models and systems.

Model implementations and their search spaces live together in each package.
Importing this package loads each public child module or package; shared helpers are
excluded. Optional backends load during model execution.
"""

import importlib
import pkgutil


def _register_builtin_models() -> None:
    """Import baseline modules and packages.

    Propagate failures from their implementations.
    """
    for _finder, name, _is_package in pkgutil.iter_modules(__path__):
        if not name.startswith("_"):
            importlib.import_module(f"{__name__}.{name}")


_register_builtin_models()
