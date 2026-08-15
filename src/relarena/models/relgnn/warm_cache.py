"""Public RelGNN graph warmer, runnable with `python -m`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from relarena.cache import resolve_cache_config
from relarena.dataset import RelBenchDatasetTask
from relarena.models.relgnn.preprocessing import load_graph


def precompute_dataset_task(
    dataset_name: str,
    task_name: str,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    device_ids: list[torch.device] | None = None,
) -> str:
    """Fill both phase graphs; task is used only to load the dataset splits."""
    del device_ids
    cache = resolve_cache_config(cache_dir, on_miss="fill")
    if cache.directory is None:
        raise ValueError("RelGNN warming needs --cache-dir or RELARENA_CACHE_DIR")
    source = RelBenchDatasetTask(dataset_name, task_name, download=download)
    for phase, split in (
        ("inner", source.inner_split()),
        ("outer", source.outer_split()),
    ):
        load_graph(
            split.db_state,
            cache,
            source.run_identity(phase),
        )
    return f"{dataset_name}/{task_name}"


def main(argv: list[str] | None = None) -> int:
    """Warm one native RelBench dataset's inner and outer graphs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    precompute_dataset_task(
        args.dataset,
        args.task,
        cache_dir=args.cache_dir,
        download=not args.no_download,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
