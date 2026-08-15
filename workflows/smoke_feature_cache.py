"""Manual CPU smoke check: fill the DFS cache, then verify real read hits.

Builds the store through the shared DFS warmer on a real RelBench task, then
strictly reads every canonical request: inner train history, outer train-only
history (RDBLearn), and outer train+val history (full-refit models). Read mode
raises on any miss, so completing the second pass proves that the warmer covers
both final-fit protocols.

No TFM or GPU is involved. Run:

    OMP_NUM_THREADS=1 uv run --extra rdblearn python workflows/smoke_feature_cache.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import relarena.featurization.dfs as dfs_mod
from relarena.cache import CacheConfig
from relarena.dataset import RelBenchDatasetTask, concat_tables
from relarena.featurization import DFS_MAX_DEPTH, build_dfs_features
from relarena.featurization.warm_cache import warm_dfs_cache

DATASET, TASK = "rel-f1", "driver-dnf"


def _reset_memo() -> None:
    """Clear the process-local DFS memo so reads must come from disk."""
    dfs_mod._CACHE = dfs_mod._DepthCache()


def main() -> None:
    """Fill and read-verify the cache for one small RelBench task."""
    source = RelBenchDatasetTask(DATASET, TASK)
    cache_dir = Path(tempfile.mkdtemp(prefix="relarena_cache_smoke_")) / "cache"
    print(f"task: {DATASET}/{TASK}")
    print(f"cache dir: {cache_dir}\n")

    _reset_memo()
    started = time.perf_counter()
    warm_dfs_cache(source, CacheConfig(cache_dir, "fill"))
    build_seconds = time.perf_counter() - started
    artifacts = sorted(cache_dir.rglob("*.parquet"))
    print(
        f"[fill]  precompute: {build_seconds:.1f}s -> "
        f"{len(artifacts)} content-keyed artifacts:"
    )
    for artifact in artifacts:
        print(f"          {artifact.name}")
    assert artifacts, "precompute wrote no artifacts"

    inner, outer = source.inner_split(), source.outer_split()
    full_outer_history = concat_tables(outer.train_table, outer.val_table)
    _reset_memo()
    started = time.perf_counter()
    for phase, db, history, evaluation in (
        ("inner", inner.db_state, inner.train_table, inner.eval_table),
        ("outer", outer.db_state, outer.train_table, outer.eval_table),
        ("outer", outer.db_state, full_outer_history, outer.eval_table),
    ):
        for anchors in (history, evaluation):
            build_dfs_features(
                source.task,
                db,
                anchors,
                depth=DFS_MAX_DEPTH,
                max_depth=DFS_MAX_DEPTH,
                history_table=history if source.task.time_col else None,
                cache=CacheConfig(cache_dir, "raise"),
                run_identity=source.run_identity(phase),
            )
    read_seconds = time.perf_counter() - started
    print(
        f"\n[read]  fit featurization: {read_seconds:.1f}s, "
        "every artifact hit (no CacheMiss)"
    )
    assert read_seconds < build_seconds, "read was not faster than fill"
    print(
        f"        speedup: {build_seconds:.1f}s fill -> {read_seconds:.1f}s read "
        f"({build_seconds / read_seconds:.0f}x)"
    )


if __name__ == "__main__":
    main()
