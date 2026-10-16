"""Shared DFS cache warming for supplied task sources."""

from __future__ import annotations

from collections.abc import Sequence

from relarena_core.cache import CacheConfig
from relarena_core.dataset import TaskSource, concat_tables
from relarena_core.featurization.dfs import DFS_MAX_DEPTH, build_dfs_features

#: Independent label-history regimes, so a warm run can fan out one job per task and
#: group.
HISTORY_GROUPS = ("inner", "outer-train", "outer-full")


def warm_dfs_cache(
    source: TaskSource,
    cache: CacheConfig,
    *,
    max_depth: int = DFS_MAX_DEPTH,
    groups: Sequence[str] = HISTORY_GROUPS,
) -> None:
    """Fill shared DFS matrices for tuning and both final-fit history regimes."""
    if cache.directory is None or cache.on_miss != "fill":
        raise ValueError("DFS warming needs CacheConfig(directory, on_miss='fill')")
    unknown = set(groups) - set(HISTORY_GROUPS)
    if unknown:
        raise ValueError(f"Unknown DFS history groups {sorted(unknown)}")
    phases = []
    if "inner" in groups:
        inner = source.inner_split()
        phases.append(("inner", inner.db_state, inner.train_table, inner.eval_table))
    if "outer-train" in groups or "outer-full" in groups:
        outer = source.outer_split()
        # RDBLearn follows its published train-only final-fit protocol, whereas
        # TabPFN-Rel refits on train+val. The actual history input
        # belongs in the key, so warm both canonical regimes without model dispatch.
        if "outer-train" in groups:
            phases.append(
                ("outer", outer.db_state, outer.train_table, outer.eval_table)
            )
        if "outer-full" in groups:
            history = concat_tables(outer.train_table, outer.val_table)
            phases.append(("outer", outer.db_state, history, outer.eval_table))
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
