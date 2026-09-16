"""Public RelGT token warmer, runnable with `python -m`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from relarena.cache import resolve_cache_config
from relarena.dataset import RelBenchDatasetTask
from relarena.models._shared.gnn.graph import build_graph

_NUM_NEIGHBORS = 300


def _load_precompute_tokens() -> Callable[..., None]:
    """Load RelGT's token writer only when the optional model extra is in use."""
    from relarena.models.relgt.tokenize import precompute_tokens

    return precompute_tokens


def precompute_dataset_task(
    dataset_name: str,
    task_name: str,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    num_workers: int = 0,
) -> str:
    """Fill every inner/outer token artifact used by RelGT fit and predict."""
    cache = resolve_cache_config(cache_dir, on_miss="fill")
    if cache.directory is None:
        raise ValueError("RelGT warming needs --cache-dir or RELARENA_CACHE_DIR")
    source = RelBenchDatasetTask(dataset_name, task_name, download=download)
    precompute_tokens = _load_precompute_tokens()
    for phase, split in (
        ("inner", source.inner_split()),
        ("outer", source.outer_split()),
    ):
        data, _ = build_graph(split.db_state, "cpu")
        tables = [split.train_table, split.eval_table]
        val_table = getattr(split, "val_table", None)
        if val_table is not None:
            tables.append(val_table)
        precompute_tokens(
            data,
            source.task,
            tables,
            _NUM_NEIGHBORS,
            cache,
            run_identity=source.run_identity(phase),
            num_workers=num_workers,
        )
    return f"{dataset_name}/{task_name}"


def main(argv: list[str] | None = None) -> int:
    """Warm one native RelBench task's inner and outer token files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    precompute_dataset_task(
        args.dataset,
        args.task,
        cache_dir=args.cache_dir,
        download=not args.no_download,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
