"""Benchmark integration with model discovery."""

import pytest

from relarena.evaluation import leaderboard
from relarena_core.registry import MethodRegistry
from relarena_core.system import RelArenaSystem


def test_leaderboard_discovers_external_system_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated = MethodRegistry()
    system = type("ExternalSystem", (RelArenaSystem,), {"name": "external-system"})

    def discover() -> None:
        isolated.register_system(system)

    monkeypatch.setattr(leaderboard, "registry", isolated)
    monkeypatch.setattr(leaderboard, "discover_models", discover)
    assert leaderboard.method_kind("external-system") == "system"
