"""Tests for relarena.models._shared.gnn.graph_cache."""

from relarena.models._shared.gnn.graph_cache import DBGraphCache


def test__get__same_db__builds_once_and_reuses() -> None:
    """Repeated get() for one db builds once and returns the same memoized object."""
    cache = DBGraphCache()
    db = object()
    calls: list[int] = []

    def build() -> tuple[dict[str, int], dict[str, bool]]:
        calls.append(1)
        return ({"data": len(calls)}, {"stats": True})

    first = cache.get(db, build)
    second = cache.get(db, build)

    assert calls == [1]  # built exactly once
    assert first is second  # second call reused the memoized value


def test__get__different_db__rebuilds() -> None:
    """A different db identity rebuilds — the cache holds one graph at a time."""
    cache = DBGraphCache()
    db1, db2 = object(), object()  # both kept alive so id() can't be reused
    calls: list[int] = []

    def build() -> tuple[int, dict[str, int]]:
        calls.append(1)
        return (len(calls), {})

    cache.get(db1, build)
    cache.get(db2, build)  # evicts db1
    again = cache.get(db1, build)  # db1 no longer cached -> rebuild

    assert calls == [1, 1, 1]
    assert again[0] == 3  # the third build's value


def test__get__same_db_different_variant__rebuilds() -> None:
    cache = DBGraphCache()
    db = object()
    calls: list[int] = []

    def build() -> tuple[int, dict]:
        calls.append(1)
        return len(calls), {}

    cache.get(db, build, variant="compute")
    cache.get(db, build, variant="fill")
    assert calls == [1, 1]
