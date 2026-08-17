"""Tests for relarena.models._shared.gnn.graph_cache."""

from relarena.models._shared.gnn.graph_cache import DBGraphCache


class _DB:
    """Stand-in for a relbench `Database` (same-size objects reuse addresses)."""


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


def test__get__db_released_then_new_db__never_serves_the_stale_graph() -> None:
    """A released db's address, reused by the next one, must not be a false hit.

    Mirrors `run_experiment`: the inner split's db goes unreferenced when `tune`
    returns, then the outer split allocates a new one onto the freed address. A
    hit there would score the final test fit on the val-censored graph.
    """
    cache = DBGraphCache()
    for i in range(100):
        cache.get(_DB(), lambda: ("inner", {}))  # unreferenced once get() returns
        assert cache.get(_DB(), lambda: (f"outer-{i}", {}))[0] == f"outer-{i}"
