"""Tests for the search-space provider mechanism (static space vs factory)."""

from __future__ import annotations

from relarena.search_space import (
    SearchSpace,
    TaskStats,
    resolve_search_space,
)


def test__resolve_search_space__static_space__returned_unchanged() -> None:
    space = SearchSpace(default_overrides={"a": 1})
    # A SearchSpace is not callable, so resolution is a no-op (the common case).
    assert resolve_search_space(space, TaskStats(num_train_nodes=10)) is space


def test__resolve_search_space__factory__called_with_stats() -> None:
    seen: list[int] = []

    def factory(stats: TaskStats) -> SearchSpace:
        seen.append(stats.num_train_nodes)
        return SearchSpace(default_overrides={"n": stats.num_train_nodes})

    resolved = resolve_search_space(factory, TaskStats(num_train_nodes=42))
    assert seen == [42]
    assert resolved.default_overrides == {"n": 42}
