"""Warm shared DFS caches for named benchmark tasks with `python -m`."""

from __future__ import annotations

import argparse
from pathlib import Path

from relarena.core.cache import resolve_cache_config
from relarena.core.featurization.dfs import DFS_MAX_DEPTH
from relarena.core.featurization.warm_cache import warm_dfs_cache
from relarena.dataset import RelBenchDatasetTask


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
