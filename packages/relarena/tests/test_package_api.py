"""Public package APIs share the runtime contracts and model registry."""

import ast
from pathlib import Path

import pytest

import relarena
import relarena_core
from relarena.userdb import PredictiveQuery
from relarena_core.userdb import PredictiveQuery as CoreQuery


@pytest.mark.parametrize(
    "name",
    [
        "RelArenaModel",
        "RelArenaSystem",
        "RunIdentity",
        "InnerSplit",
        "OuterSplit",
        "Split",
        "MethodRegistry",
        "ModelRegistry",
        "registry",
        "TrialResult",
        "SystemResult",
    ],
)
def test_public_class_and_registry_identity(name: str) -> None:
    assert getattr(relarena, name) is getattr(relarena_core, name)
    assert PredictiveQuery is CoreQuery


def test_installed_model_registration() -> None:
    model = pytest.importorskip("tabpfn_rel")
    relarena.discover_models()
    relarena.discover_models()
    assert relarena_core.registry.get("tabpfn-rel-local") is model.TabPFNRelLocalModel


@pytest.mark.parametrize("consumer", ["relarena", "tabpfn_rel"])
def test_consumers_use_public_core_interfaces(consumer: str) -> None:
    package = pytest.importorskip(consumer)
    violations = []
    for path in Path(package.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
                if node.module.startswith("relarena_core"):
                    if any(name.name.startswith("_") for name in node.names):
                        violations.append(f"{path}:{node.lineno}: private core name")
            elif isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            for module in modules:
                if module.startswith("relarena_core") and any(
                    part.startswith("_") for part in module.split(".")
                ):
                    violations.append(f"{path}:{node.lineno}: {module}")
    assert not violations, "\n".join(violations)
