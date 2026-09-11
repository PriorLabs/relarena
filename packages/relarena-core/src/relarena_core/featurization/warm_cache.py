"""Shared DFS cache warming for supplied task sources."""

from __future__ import annotations

from relarena_core.cache import CacheConfig
from relarena_core.dataset import TaskSource, concat_tables
from relarena_core.featurization.dfs import DFS_MAX_DEPTH, build_dfs_features


def warm_dfs_cache(
    source: TaskSource,
    cache: CacheConfig,
    *,
    max_depth: int = DFS_MAX_DEPTH,
) -> None:
    """Fill shared DFS matrices for tuning and both final-fit history regimes."""
    if cache.directory is None or cache.on_miss != "fill":
        raise ValueError("DFS warming needs CacheConfig(directory, on_miss='fill')")
    inner, outer = source.inner_split(), source.outer_split()
    full_outer_history = concat_tables(outer.train_table, outer.val_table)
    phases = (
        ("inner", inner.db_state, inner.train_table, inner.eval_table),
        # RDBLearn follows its published train-only final-fit protocol, whereas
        # TabPFN-Rel refits on train+val. The actual history input
        # belongs in the key, so warm both canonical regimes without model dispatch.
        ("outer", outer.db_state, outer.train_table, outer.eval_table),
        (
            "outer",
            outer.db_state,
            full_outer_history,
            outer.eval_table,
        ),
    )
    for phase, db, history, evaluation in phases:
        identity = source.run_identity(phase)
        for anchors in (history, evaluation):
            build_dfs_features(
                source.task,
                db,
                anchors,
                depth=max_depth,
                max_depth=max_depth,
                history_table=history if source.task.time_col else None,
                cache=cache,
                run_identity=identity,
            )
