"""Tests for the model auto-discovery in `relarena.models`.

That each model lands in the registry is covered by the per-model tests; these
cover the scan itself — what it skips, and which import failures it tolerates.
The `importlib` name is stubbed inside the package namespace rather than patching
the real module, so nothing else importing during the test is affected.
"""

from __future__ import annotations

import json
import logging
import pkgutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import relarena.models as models_pkg


def test__import_models__does_not_import_preprocessing_warmers() -> None:
    """Model registration must not require any preprocessing-format extra."""
    code = (
        "import sys; import relarena.models; "
        "assert 'relarena.models.relgnn.warm_cache' not in sys.modules; "
        "assert 'relarena.models.relgt.warm_cache' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


_REGISTRY_NAMES = """
import json
import sys

if "--without-dfs-extra" in sys.argv:

    class _Hidden:
        '''Make `fastdfs` unimportable, as on an install without the DFS extra.'''

        def find_spec(self, name, path=None, target=None):
            if name == "fastdfs" or name.startswith("fastdfs."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _Hidden())

import relarena.models  # noqa: F401  - importing runs the registration scan
from relarena import registry

print(json.dumps(sorted(registry.names())))
"""


def _registered_names(*args: str) -> list[str]:
    """Names in the registry of a fresh interpreter, run with `args`."""
    result = subprocess.run(
        [sys.executable, "-c", _REGISTRY_NAMES, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test__register_builtin_models__without_the_dfs_extra__same_models() -> None:
    """Registration is dep-free: an absent extra must not drop a model.

    The dev group installs `fastdfs`, so a module-scope import of it is invisible
    to the rest of the suite — it surfaces only on an install without the DFS
    extra, where the scan reads the ImportError as an absent optional dep and the
    model goes quietly missing.
    """
    assert _registered_names("--without-dfs-extra") == _registered_names()


def _record_imports(monkeypatch: pytest.MonkeyPatch, fail: dict[str, str]) -> list[str]:
    """Stub out module importing; return the list of names the scan asks for.

    `fail` maps a model name to the missing-module name its import should raise
    `ModuleNotFoundError` for.
    """
    asked: list[str] = []

    def _import(target: str) -> None:
        name = target.rsplit(".", 1)[-1]
        asked.append(name)
        if name in fail:
            raise ModuleNotFoundError(
                f"No module named {fail[name]!r}", name=fail[name]
            )

    monkeypatch.setattr(models_pkg, "importlib", SimpleNamespace(import_module=_import))
    return asked


def test__register_builtin_models__scans_models_and_skips_private_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _record_imports(monkeypatch, fail={})
    models_pkg._register_builtin_models()

    # The real package layout is walked, so this pins the actual exclusions:
    # shared helpers and vendored upstream code must not be imported as models.
    assert "lightgbm" in asked
    assert not [name for name in asked if name.startswith("_")]
    assert "_shared" not in asked


def test__register_builtin_models__missing_third_party_dep__skips_that_model(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An absent per-model extra is a normal install, not a broken one. Fail the
    # first model the scan reaches and assert a *later* one still gets imported:
    # asserting on an earlier name would pass even if a skip aborted the loop.
    scanned = [
        name
        for _finder, name, _is_pkg in pkgutil.iter_modules(models_pkg.__path__)
        if not name.startswith("_")
    ]
    first, last = scanned[0], scanned[-1]
    asked = _record_imports(monkeypatch, fail={first: "a_missing_extra"})

    with caplog.at_level(logging.INFO, logger=models_pkg.__name__):
        models_pkg._register_builtin_models()

    assert first in asked  # attempted
    assert last in asked  # and the scan carried on past the failure
    assert f"Skipping model {first!r}" in caplog.text


def test__register_builtin_models__missing_relarena_module__raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing internal module is a defect; swallowing it would drop a model
    # from the registry silently.
    _record_imports(monkeypatch, fail={"dummy": "relarena.does_not_exist"})

    with pytest.raises(ModuleNotFoundError, match="relarena.does_not_exist"):
        models_pkg._register_builtin_models()
