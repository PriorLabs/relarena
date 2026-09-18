"""Package imports and installed entry-point registration."""

from __future__ import annotations

import subprocess
import sys

import pytest
from relarena_core.registry import registry
from relarena_core.userdb import PredictiveQuery as ArenaQuery

from tabpfn_rel import PredictiveQuery, TabPFNRelLocalModel


@pytest.mark.parametrize("first", ["relarena_core", "tabpfn_rel"])
def test_import_orders_and_repeated_discovery(first: str) -> None:
    code = f"""
import sys
import {first}
import relarena_core
import tabpfn_rel
assert set(relarena_core.registry.names()) == {{
    'tabpfn-rel-local', 'tabpfn-rel-client'
}}
assert 'relarena.models' not in sys.modules
for name in ('tabpfn', 'tabpfn_client', 'fastdfs'):
    assert name not in sys.modules, name
relarena_core.discover_models()
relarena_core.discover_models()
local = relarena_core.registry.get('tabpfn-rel-local')
assert local is tabpfn_rel.TabPFNRelLocalModel
client = relarena_core.registry.get('tabpfn-rel-client')
assert client is tabpfn_rel.TabPFNRelClientModel
assert 'tabpfn' not in sys.modules
assert 'tabpfn_client' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_decorator_registration_and_shared_rpi() -> None:
    assert registry.get("tabpfn-rel-local") is TabPFNRelLocalModel
    assert PredictiveQuery is ArenaQuery
