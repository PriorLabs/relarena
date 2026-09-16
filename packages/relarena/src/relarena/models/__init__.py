"""Concrete model wrappers.

Importing this package registers every built-in model into the global registry:
each wrapper module is imported in turn, and the `@register_model` decorator on
its class does the registering. A wrapper implements the
`relarena.model.RelArenaModel` contract (`fit` / `predict`) and declares a
`relarena.search_space.SearchSpace`; see `lightgbm.py` for a worked example.

Adding a model needs no edit here — drop a module or package alongside the
existing ones and it is picked up. Names starting with an underscore are skipped,
which keeps `_shared` and the vendored packages out of the scan.

Reach a model through the registry, by name; the wrapper classes and preprocessing
warmers are not exported here. Each cache owner exposes its runnable warmer from its
own `warm_cache` module, so adding one never changes this discovery package.
"""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def _register_builtin_models() -> None:
    """Import every wrapper module so its `@register_model` decorator runs.

    A wrapper whose *third-party* dependency is absent is skipped: the per-model
    extras are optional, so `tabpfn-rel` without the DFS deps is a normal install
    rather than a broken one. A missing `relarena` module, or any other import
    error, is a defect and propagates — swallowing those is how a model goes
    quietly missing from the registry.
    """
    for _finder, name, _is_pkg in pkgutil.iter_modules(__path__):
        if name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.split(".")[0] == "relarena":
                raise
            logger.info(
                "Skipping model %r: optional dependency %r is not installed.",
                name,
                exc.name,
            )


_register_builtin_models()
