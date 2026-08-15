"""Example: TabPFN-Rel fit() with and without a precomputed feature cache.

Before the model runs, TabPFN-Rel's expensive step is Deep Feature Synthesis (DFS):
flattening the relational database into a feature table. That featurization depends
only on the (censored) data, not on the TFM or the seed, so it can be computed once
and reused. Pass an explicit `CacheConfig(directory, "raise")` to make the model
read DFS features from that store; a missing artifact raises, so the store is built
up front by `warm_dfs_cache` below. Use `CacheConfig(None, "compute")` to run
without a persistent cache.

This script fits the same model twice on one RelBench task -- once with no cache, and
once against a store warmed up front by `warm_dfs_cache` -- and checks the outputs
are identical (the cache only changes speed). The uncached DFS is what the cache
saves; the TabPFN forward pass runs either way and is what needs a GPU.

Run (GPU recommended; the first run downloads the RelBench dataset):

    uv run --extra rdblearn python examples/tabpfn_rel_caching.py

RELARENA_EXAMPLE_SKIP_TFM=1 runs only the DFS featurization + cache and skips the
TabPFN forward pass, so the caching can be exercised on CPU / locally with no GPU. On
macOS also set OMP_NUM_THREADS=1 (torch/lightgbm libomp).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from relarena.cache import CacheConfig
from relarena.dataset import OuterSplit, RelBenchDatasetTask, concat_tables
from relarena.featurization import build_dfs_features
from relarena.featurization import dfs as dfs_mod
from relarena.featurization.warm_cache import warm_dfs_cache
from relarena.models._shared.tfm.tfm import default_device
from relarena.models.tabpfn_rel.model import TABPFN_REL_LOCAL_SPACE, TabPFNRelModel

#: A reasonably sized RelBench entity task: small enough to run, big enough that the
#: DFS cost is visible. Swap for e.g. ("rel-hm", "user-churn") for a heavier one.
DATASET, TASK = "rel-f1", "driver-dnf"
SEED = 0

#: Debug: skip the TabPFN forward pass and run only the DFS featurization + cache
#: (CPU-only, no GPU), so the caching can be exercised locally. Default off.
DEBUG_SKIP_TFM = os.environ.get("RELARENA_EXAMPLE_SKIP_TFM", "") == "1"

#: The reference TabPFN-Rel config, minus text embeddings (keeps dependencies to the
#: `rdblearn` extra; the DFS cache is what this example is about).
CONFIG = {**TABPFN_REL_LOCAL_SPACE.default_overrides}


def _reset_in_process_cache() -> None:
    """Clear the module-level in-process DFS memo.

    Within one process TabPFN-Rel memoizes the DFS matrix by object identity, so a
    second fit on the same split would reuse it regardless of any on-disk cache.
    Resetting it before each run simulates a fresh process, so the timings reflect
    only the persisted cache: the realistic precompute-on-CPU / fit-on-GPU case.
    """
    dfs_mod._CACHE = dfs_mod._DepthCache()


def _fit_and_predict(
    src: RelBenchDatasetTask, split: OuterSplit, cache_dir: str | None
) -> tuple[np.ndarray, float]:
    """Fit on the split's train, predict its eval; return (preds, seconds)."""
    _reset_in_process_cache()
    cache = (
        CacheConfig(None, "compute")
        if cache_dir is None
        else CacheConfig(Path(cache_dir), "raise")
    )
    model = TabPFNRelModel(CONFIG, cache=cache, run_identity=src.run_identity("outer"))
    history = concat_tables(split.train_table, split.val_table)
    start = time.perf_counter()
    model.fit(src.task, split.db_state, history, None, seed=SEED)
    preds = model.predict(src.task, split.db_state, split.eval_table)
    return preds, time.perf_counter() - start


def _featurize(
    src: RelBenchDatasetTask, split: OuterSplit, cache_dir: str | None
) -> tuple[pd.DataFrame, float]:
    """DFS featurization only (no TFM) under the given cache; returns (features, secs).

    This is exactly the featurization `fit` does internally, minus the TabPFN forward
    pass -- so DEBUG_SKIP_TFM mode exercises the cache on CPU with no GPU.
    """
    _reset_in_process_cache()
    # Match the outer refit exactly (same args -> same keys as warm_dfs_cache).
    anchors = concat_tables(split.train_table, split.val_table)
    history = anchors if src.task.time_col else None
    cache = (
        CacheConfig(None, "compute")
        if cache_dir is None
        else CacheConfig(Path(cache_dir), "raise")
    )
    start = time.perf_counter()
    feats, _ = build_dfs_features(
        src.task,
        split.db_state,
        anchors,
        depth=int(CONFIG["max_depth"]),
        max_depth=TabPFNRelModel.MAX_DEPTH,
        history_table=history,
        keep_anchor_columns=True,
        cache=cache,
        run_identity=src.run_identity("outer"),
    )
    return feats, time.perf_counter() - start


def _precompute_local(src: RelBenchDatasetTask, split: OuterSplit) -> str:
    """Warm a fresh temp dir with the DFS features this run needs (no TFM, CPU)."""
    _reset_in_process_cache()  # else a prior fit's in-memory memo hides the disk write
    cache_dir = str(Path(tempfile.mkdtemp(prefix="tabpfn_rel_cache_")) / "cache")
    # Fill mode computes the canonical DFS artifacts and writes them to the store.
    warm_dfs_cache(
        src,
        CacheConfig(Path(cache_dir), "fill"),
        max_depth=TabPFNRelModel.MAX_DEPTH,
    )
    n = len(list(Path(cache_dir).rglob("*.parquet")))
    print(f"      precomputed {n} parquet artifact(s) at {cache_dir}")
    return cache_dir


def main() -> None:
    """Run twice (no cache, then a precomputed cache) and check the outputs match."""
    print(f"Loading {DATASET}/{TASK} (downloads on first run) ...")
    src = RelBenchDatasetTask(DATASET, TASK)
    split = src.outer_split()
    print(
        f"      split: {len(split.train_table.df)} train rows, "
        f"{len(split.eval_table.df)} eval rows"
    )
    if DEBUG_SKIP_TFM:
        print("      DEBUG_SKIP_TFM: DFS featurization + cache only (no GPU/TFM)")
    else:
        print(f"      TabPFN device: {default_device()}")

    run = _featurize if DEBUG_SKIP_TFM else _fit_and_predict
    label = "DFS featurization" if DEBUG_SKIP_TFM else "fit + predict"

    print(f"\n[1/2] {label} WITHOUT a cache: DFS is computed ...")
    out_nocache, t_nocache = run(src, split, None)
    print(f"      {t_nocache:.1f}s")

    print(f"\n[2/2] {label} WITH a precomputed cache: DFS is read from disk ...")
    cache_dir = _precompute_local(src, split)
    out_cache, t_cache = run(src, split, cache_dir)
    print(f"      {t_cache:.1f}s")

    print(
        f"\nDFS saved ~{t_nocache - t_cache:.1f}s ({t_nocache:.1f}s -> {t_cache:.1f}s)."
    )
    if DEBUG_SKIP_TFM:
        # The feature matrix mixes numeric and categorical columns, so compare frames
        # (check_dtype=False tolerates the parquet round-trip's dtype normalization).
        pd.testing.assert_frame_equal(out_nocache, out_cache, check_dtype=False)
        print("DFS features are identical with and without the cache (TFM skipped).")
    else:
        assert np.allclose(out_nocache, out_cache, equal_nan=True), (
            "cached and uncached predictions must match; the cache only affects speed"
        )
        print("Predictions are identical with and without the cache.")


if __name__ == "__main__":
    main()
