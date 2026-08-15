"""Pre-build the feature cache (DFS matrices) before an eval sweep.

The feature cache is read-only during a run: with `RELARENA_CACHE_DIR` set,
a missing artifact raises rather than recompute (so a
sweep never quietly pays the featurization cost). This script is the *fill* side —
it computes and writes every artifact the eval will later read, on CPU, once.
Text is not part of the cache: the TFM's estimator embeds it at fit time.

For each `(dataset, task)` it invokes the shared DFS module's public warmer. It
builds the inner and outer full-anchor, leak-safe-history matrices consumed by every
DFS model; no TFM is involved.

**Where the cache lives.** The store is the directory given by
`RELARENA_CACHE_DIR`; point the warm-up and the eval at the same one. It is
idempotent: artifacts already present are skipped. Bump the DFS cache version when
algorithm changes need new immutable artifacts.

Run from the repository root:

    RELARENA_CACHE_DIR=~/relarena_features \
        uv run --extra rdblearn python workflows/warm_feature_cache.py

CPU is enough; run it ahead of a (GPU) eval pointing at the same store.
"""

from __future__ import annotations

import argparse
import os
import sys

from relarena.cache import resolve_cache_config
from relarena.dataset import RelBenchDatasetTask
from relarena.featurization.warm_cache import warm_dfs_cache
from relarena.tasks import RELBENCH_V1_DATASETS, list_entity_tasks


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(RELBENCH_V1_DATASETS),
        help="datasets to warm (default: all RelBench v1)",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="restrict to these task names (default: every entity task)",
    )
    p.add_argument("--no-download", action="store_true", help="assume datasets cached")
    return p.parse_args(sys.argv[1:] if argv is None else argv)


def warm_task(dataset: str, task_name: str, *, download: bool) -> None:
    """Warm the inner + outer split caches for one `(dataset, task)`."""
    src = RelBenchDatasetTask(dataset, task_name, download=download)
    warm_dfs_cache(src, resolve_cache_config(None, on_miss="fill"))


def main(argv: list[str] | None = None) -> int:
    """Warm the feature cache for the configured `(dataset, task)` pairs."""
    args = _parse_args(argv)
    if not os.environ.get("RELARENA_CACHE_DIR"):
        print(
            "RELARENA_CACHE_DIR not set — nothing to fill.",
            file=sys.stderr,
        )
        return 2

    specs = list_entity_tasks(args.datasets)
    if args.tasks is not None:
        wanted = set(args.tasks)
        specs = [s for s in specs if s.task in wanted]

    print(f"warming {len(specs)} task(s)")
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.dataset}/{spec.task}", flush=True)
        warm_task(spec.dataset, spec.task, download=not args.no_download)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
