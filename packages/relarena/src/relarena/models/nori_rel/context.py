"""Bounded training reservoirs and the cache-safe inference window."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from relbench.base import EntityTask, Table

MAX_CONTEXT_ROWS: Final = 60_000
TEMPORAL_RESERVOIR_RECENT_ROWS: Final = MAX_CONTEXT_ROWS // 2
#: Nori's forward pass holds context plus query rows; keep the sum under this.
MAX_FORWARD_ROWS: Final = 64_000
CACHE_TRIGGER_MARGIN_ROWS: Final = 2
RESERVOIR_RANDOM: Final = "random"
RESERVOIR_TEMPORAL_MIX: Final = "temporal_mix_v1"
RESERVOIRS: Final = frozenset({RESERVOIR_RANDOM, RESERVOIR_TEMPORAL_MIX})


def random_context_indices(
    n_rows: int, seed: int, *, cap: int | None = None
) -> np.ndarray:
    """Keep every row up to the cap, otherwise a seeded random subset."""
    limit = MAX_CONTEXT_ROWS if cap is None else cap
    if n_rows <= limit:
        return np.arange(n_rows)
    return np.random.default_rng(seed).permutation(n_rows)[:limit]


def temporal_context_indices(task: EntityTask, table: Table, seed: int) -> np.ndarray:
    """Mix each entity's latest row with seeded random coverage of the rest."""
    n_rows = len(table.df)
    if n_rows <= MAX_CONTEXT_ROWS:
        return np.arange(n_rows)

    time_column = getattr(task, "time_col", None)
    if not isinstance(time_column, str) or time_column not in table.df:
        return np.sort(random_context_indices(n_rows, seed))
    times = pd.to_datetime(
        table.df[time_column], errors="coerce", utc=True, format="mixed"
    ).reset_index(drop=True)
    valid_time = times.notna().to_numpy()
    if not valid_time.any():
        return np.sort(random_context_indices(n_rows, seed))

    positions = np.arange(n_rows, dtype=np.int64)
    candidates = pd.DataFrame({"time": times, "position": positions}).loc[valid_time]
    entity_column = getattr(task, "entity_col", None)
    if isinstance(entity_column, str) and entity_column in table.df:
        candidates["entity"] = table.df[entity_column].to_numpy()[valid_time]
        candidates = candidates.loc[candidates["entity"].notna()]
        if not candidates.empty:
            latest = candidates.groupby("entity", sort=False, observed=True)[
                "time"
            ].idxmax()
            candidates = candidates.loc[latest]

    rng = np.random.default_rng(seed)
    candidates = candidates.assign(tie=rng.random(len(candidates))).sort_values(
        ["time", "tie", "position"], ascending=[False, True, True], kind="mergesort"
    )
    recent_count = min(TEMPORAL_RESERVOIR_RECENT_ROWS, len(candidates))
    recent = candidates["position"].to_numpy(dtype=np.int64)[:recent_count]
    available = np.ones(n_rows, dtype=bool)
    available[recent] = False
    coverage = rng.permutation(positions[available])[: MAX_CONTEXT_ROWS - len(recent)]
    return np.sort(np.concatenate([recent, coverage]))


def training_reservoir(
    task: EntityTask, table: Table, seed: int, policy: str
) -> np.ndarray:
    """Select the bounded training rows under one named policy."""
    if policy == RESERVOIR_RANDOM:
        return random_context_indices(len(table.df), seed)
    if policy == RESERVOIR_TEMPORAL_MIX:
        return temporal_context_indices(task, table, seed)
    raise ValueError(f"unknown training reservoir {policy!r}")


def cache_safe_random_window(problem: Any, rng: np.random.Generator) -> np.ndarray:
    """Draw Nori's random context window while bounding the forward pass."""
    pool_size = min(problem.window, MAX_CONTEXT_ROWS, MAX_FORWARD_ROWS - 1)
    pool = rng.permutation(problem.n_train)[:pool_size]
    max_query_rows = max(
        1, min(MAX_FORWARD_ROWS - pool_size, problem.window + CACHE_TRIGGER_MARGIN_ROWS)
    )
    if problem.n_test > max_query_rows:
        problem.query_chunk = max_query_rows
        return problem.predict(pool)

    query = np.arange(problem.n_test)
    if (
        problem.n_test <= problem.window
        and problem.window + CACHE_TRIGGER_MARGIN_ROWS <= max_query_rows
    ):
        # Nori opens its cache above one window. Query rows are independent, so
        # repeated indices only trigger that path; their outputs are discarded.
        query = np.resize(query, problem.window + CACHE_TRIGGER_MARGIN_ROWS)
    problem.query_chunk = len(query)
    return problem.predict(pool, query_idx=query)[: problem.n_test]


__all__ = [
    "MAX_CONTEXT_ROWS",
    "RESERVOIRS",
    "RESERVOIR_RANDOM",
    "RESERVOIR_TEMPORAL_MIX",
    "cache_safe_random_window",
    "random_context_indices",
    "temporal_context_indices",
    "training_reservoir",
]
