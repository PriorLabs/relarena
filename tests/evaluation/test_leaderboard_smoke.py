"""Smoke test for the optional `leaderboard` extra (bencheval).

Skipped unless the `leaderboard` extra is installed (`uv sync --extra
leaderboard`); CI installs that extra explicitly, so this guards that the pinned
bencheval git dependency keeps importing.
"""

from __future__ import annotations

import pytest

bencheval_evaluator = pytest.importorskip("bencheval.evaluator")


def test_bencheval_evaluator_entry_point_importable() -> None:
    # `BenchmarkEvaluator` is the entry point relarena.evaluation.leaderboard
    # builds on to compute the leaderboard from a results DataFrame.
    assert hasattr(bencheval_evaluator, "BenchmarkEvaluator")
