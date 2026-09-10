import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["relarena", "tabpfn_rel"])
def test_installed_discovery(first: str) -> None:
    code = f"""
import {first}
import sys
from importlib.metadata import entry_points
from relarena import discover_models, registry
from tabpfn_rel import TabPFNRelClientModel, TabPFNRelLocalModel
assert any(e.value == "tabpfn_rel.model" for e in entry_points(group="relarena.models"))
for _ in range(2):
    discover_models()
    assert registry.get("tabpfn-rel-local") is TabPFNRelLocalModel
    assert registry.get("tabpfn-rel-client") is TabPFNRelClientModel
assert not {{"tabpfn", "tabpfn_client", "fastdfs"}} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
