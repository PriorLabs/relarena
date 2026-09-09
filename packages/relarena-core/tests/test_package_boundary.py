"""Core imports must remain independent of benchmark and model packages."""

import subprocess
import sys


def test_imports_and_schema_loading_without_consumers() -> None:
    code = """
import importlib.abc
import pkgutil
import sys

class BlockConsumers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'relarena', 'tabpfn_rel'}:
            raise AssertionError(f'Core attempted a consumer import: {fullname}')
        return None

sys.meta_path.insert(0, BlockConsumers())
import relarena_core
for info in pkgutil.walk_packages(relarena_core.__path__, 'relarena_core.'):
    __import__(info.name)
from relarena_core.userdb._schema import load_schema
assert load_schema('database.schema.json')['type'] == 'object'
assert load_schema('task.schema.json')['type'] == 'object'
assert not relarena_core.registry.names()
assert not any(n == 'relarena' or n.startswith('relarena.') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
