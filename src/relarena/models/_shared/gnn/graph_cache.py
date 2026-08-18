"""In-process memo of one materialized graph, shared across tuning trials.

Materializing a RelBench database into a PyG hetero graph (or loading it from an
on-disk cache) is expensive and depends only on the database, not the
hyperparameters -- but the harness builds a fresh model per config. `DBGraphCache`
memoizes the built `(data, col_stats_dict)` keyed by the database object's
identity, so the tuning trials that share a censored database reuse one graph and it
rebuilds only when the database changes (e.g. the inner->outer split). Holding one
database's graph at a time is what keeps a config sweep's host-memory footprint flat
instead of growing with every reloaded graph.

The memoized database is held by reference alongside its `id()`, so a released
database cannot free its address for a later one to reuse and hit the stale entry.

Dependency-free on purpose (only `typing`): GNN models import it at module level
while keeping their heavy deps (PyG, torch_frame) lazy, so model registration still
works without the optional extras installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from relbench.base import Database


class DBGraphCache:
    """Memoize one database/variant graph (one payload at a time)."""

    def __init__(self) -> None:
        """Start empty."""
        self._key: tuple[int, object | None] | None = None
        self._value: tuple[Any, dict[str, Any]] | None = None
        self._db: Database | None = None

    def get(
        self,
        db: Database,
        build: Callable[[], tuple[Any, dict[str, Any]]],
        *,
        variant: object | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return the graph for the database identity and optional variant."""
        key = (id(db), variant)
        if key != self._key or self._value is None:
            self._value = build()
            self._key = key
            # Hold the database itself, not just its `id()`: a released database
            # frees its address for the next same-sized allocation, and the next
            # censored database landing there would be a false hit -- serving the
            # inner split's val-censored graph to the outer split's final fit.
            # Keeping the reference makes that reuse impossible. Only the current
            # entry is held, so the footprint stays one graph at a time.
            self._db = db
        return self._value
