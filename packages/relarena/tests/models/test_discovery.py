"""Automatic baseline discovery, decorator registration and backend isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_BASELINES = {
    "tabpfn-rel-local",
    "tabpfn-rel-client",
    "constant-global",
    "constant-per-entity",
    "graphsage",
    "kurversc",
    "lightgbm",
    "rdblearn",
    "relgnn",
    "relgnn-es",
    "relgt",
    "rt-plurel",
}


def test_discovery_without_optional_backends() -> None:
    code = f"""
import importlib.abc
import sys

class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{
            'fastdfs', 'tabpfn', 'tabpfn_client', 'torch_geometric',
            'torch_frame', 'relational_transformer', 'graphreduce', 'lightgbm',
        }}:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, BlockBackends())
from relarena.core.registry import registry
assert not registry.names()
import relarena.models
assert set(registry.names()) == {_BASELINES!r}
assert registry.kind('kurversc') == registry.kind('rt-plurel') == 'system'
before = list(registry)
relarena.models._register_builtin_models()
assert list(registry) == before
assert not any(name.endswith('.warm_cache') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("as_package", [False, True])
def test_new_model_is_discovered_without_a_catalogue(
    tmp_path: Path, as_package: bool
) -> None:
    package = tmp_path / "extra_model"
    if as_package:
        package.mkdir()
        module = package / "__init__.py"
    else:
        module = package.with_suffix(".py")
    module.write_text(
        "from relarena.core.model import RelArenaModel\n"
        "from relarena.core.registry import register_model\n"
        "from relarena.core.search_space import SearchSpace\n"
        "@register_model(search_space=SearchSpace(default_overrides={}))\n"
        "class ExtraModel(RelArenaModel):\n"
        "    name = 'extra-model'\n"
    )
    ignored = tmp_path / "_private_helper"
    ignored.mkdir()
    (ignored / "__init__.py").write_text(
        "raise AssertionError('private package loaded')"
    )
    code = f"""
import relarena.models
from relarena.core.registry import registry
relarena.models.__path__.append({str(tmp_path)!r})
relarena.models._register_builtin_models()
from relarena.models.extra_model import ExtraModel
assert registry.get('extra-model') is ExtraModel
assert registry.search_space('extra-model').default_overrides == {{}}
relarena.models._register_builtin_models()
assert registry.get('extra-model') is ExtraModel
assert len(registry) == 13
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("missing", ["optional_backend", "relarena.broken_internal"])
def test_import_failures_propagate(tmp_path: Path, missing: str) -> None:
    package = tmp_path / "broken_model"
    package.mkdir()
    (package / "__init__.py").write_text(
        f"raise ModuleNotFoundError('broken model import', name={missing!r})\n"
    )
    code = f"""
import relarena.models
relarena.models.__path__.append({str(tmp_path)!r})
try:
    relarena.models._register_builtin_models()
except ModuleNotFoundError as exc:
    assert exc.name == {missing!r}
else:
    raise AssertionError('Discovery hid a broken model')
"""
    subprocess.run([sys.executable, "-c", code], check=True)
