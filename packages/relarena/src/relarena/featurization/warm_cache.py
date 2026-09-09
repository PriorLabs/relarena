"""Public shared-DFS cache warmer, runnable with `python -m`."""

from __future__ import annotations

import argparse
from pathlib import Path

from relarena.cache import CacheConfig, resolve_cache_config
from relarena.dataset import RelBenchDatasetTask, concat_tables
from relarena.featurization.dfs import DFS_MAX_DEPTH, build_dfs_features


def warm_dfs_cache(
    source: RelBenchDatasetTask,
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


def main(argv: list[str] | None = None) -> int:
    """Warm one native RelBench task into a local cache directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=DFS_MAX_DEPTH)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    source = RelBenchDatasetTask(args.dataset, args.task, download=not args.no_download)
    warm_dfs_cache(
        source,
        resolve_cache_config(args.cache_dir, on_miss="fill"),
        max_depth=args.max_depth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
