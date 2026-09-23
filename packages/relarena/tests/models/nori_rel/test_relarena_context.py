"""Training reservoirs and the cache-safe inference window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table

from relarena.models.nori_rel import context as context_module
from relarena.models.nori_rel.context import (
    cache_safe_random_window,
    temporal_context_indices,
    training_reservoir,
)


def test_temporal_reservoir_mixes_recent_entities_with_random_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_module, "MAX_CONTEXT_ROWS", 4)
    monkeypatch.setattr(context_module, "TEMPORAL_RESERVOIR_RECENT_ROWS", 2)
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "cutoff": pd.date_range("2026-01-01", periods=8),
            "target": np.arange(8),
        },
        index=np.arange(100, 108),
    )
    task = SimpleNamespace(entity_col="entity", time_col="cutoff")

    first = temporal_context_indices(task, Table(frame, fkey_col_to_pkey_table={}), 7)
    second = temporal_context_indices(
        task,
        Table(frame.assign(target=np.arange(8)[::-1]), fkey_col_to_pkey_table={}),
        7,
    )

    assert len(first) == 4
    assert set(first) >= {6, 7}
    assert np.all(first[:-1] < first[1:])
    np.testing.assert_array_equal(first, second)


def test_temporal_reservoir_handles_partial_times_and_null_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_module, "MAX_CONTEXT_ROWS", 4)
    monkeypatch.setattr(context_module, "TEMPORAL_RESERVOIR_RECENT_ROWS", 2)
    table = Table(
        pd.DataFrame(
            {
                "entity": ["a", "b", None, "d", "a", "b", None, "d"],
                "cutoff": [
                    "2026-01-01",
                    None,
                    "2026-01-08",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "not-a-time",
                ],
            }
        ),
        fkey_col_to_pkey_table={},
    )
    task = SimpleNamespace(entity_col="entity", time_col="cutoff")

    actual = temporal_context_indices(task, table, 7)

    assert len(actual) == 4
    assert len(np.unique(actual)) == 4
    assert set(actual) >= {4, 5}
    assert np.all(actual[:-1] < actual[1:])


@pytest.mark.parametrize(
    ("frame", "time_col"),
    [
        (
            pd.DataFrame({"entity": range(8), "cutoff": [None, "not-a-time"] * 4}),
            "cutoff",
        ),
        (pd.DataFrame({"entity": range(8)}), None),
    ],
    ids=["invalid-times", "no-time-column"],
)
def test_temporal_reservoir_falls_back_to_seeded_random(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, time_col: str | None
) -> None:
    monkeypatch.setattr(context_module, "MAX_CONTEXT_ROWS", 3)
    task = SimpleNamespace(entity_col="entity", time_col=time_col)

    actual = training_reservoir(
        task, Table(frame, fkey_col_to_pkey_table={}), 7, "temporal_mix_v1"
    )
    expected = np.sort(np.random.default_rng(7).permutation(8)[:3])

    np.testing.assert_array_equal(actual, expected)


def test_training_reservoir_rejects_unknown_policies() -> None:
    table = Table(pd.DataFrame({"entity": range(3)}), fkey_col_to_pkey_table={})

    with pytest.raises(ValueError, match="unknown training reservoir"):
        training_reservoir(SimpleNamespace(), table, 0, "latest")


class _Problem:
    def __init__(self, n_test: int) -> None:
        self.n_train = 8
        self.window = 3
        self.n_test = n_test
        self.query_chunk = 25_000
        self.query: np.ndarray | None = None

    def predict(
        self, pool: np.ndarray, query_idx: np.ndarray | None = None
    ) -> np.ndarray:
        self.pool = pool
        self.query = query_idx
        return np.arange(self.n_test) if query_idx is None else query_idx


def test_random_window_repeats_small_queries_to_open_the_cache() -> None:
    problem = _Problem(n_test=2)
    expected_pool = np.random.default_rng(7).permutation(8)[:3]

    prediction = cache_safe_random_window(problem, np.random.default_rng(7))

    assert problem.query_chunk == 5
    np.testing.assert_array_equal(problem.pool, expected_pool)
    np.testing.assert_array_equal(problem.query, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(prediction, [0, 1])


def test_random_window_chunks_queries_to_the_forward_row_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_module, "MAX_FORWARD_ROWS", 5)
    problem = _Problem(n_test=7)

    prediction = cache_safe_random_window(problem, np.random.default_rng(7))

    assert problem.query_chunk == 2
    assert problem.query is None
    np.testing.assert_array_equal(prediction, np.arange(7))


def test_random_window_chunks_large_queries_to_one_context_window() -> None:
    problem = _Problem(n_test=7)

    prediction = cache_safe_random_window(problem, np.random.default_rng(7))

    assert problem.query_chunk == 5
    assert problem.query is None
    np.testing.assert_array_equal(prediction, np.arange(7))
