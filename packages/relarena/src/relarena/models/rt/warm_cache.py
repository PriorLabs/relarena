"""Public RT tensor-directory warmer, runnable with `python -m`.

Preprocessing is the expensive, label-free half of an RT run: rustler over the
whole censored database, then a sentence-transformer pass over every text cell.
It depends only on the database and the label tables' *rows*, not on any config,
so it is computed once here and read by every trial, seed and job afterwards —
which is what lets a configured benchmark run in `read` mode and fail loudly on
a miss instead of quietly re-embedding a database per trial.

A GPU makes the embedding step much faster but is not required.

Warms all three artifacts a run reads: the selection arm's export, the reporting
arm's refit export, and the one it scores test out of.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from relarena.cache import resolve_cache_config
from relarena.dataset import RelBenchDatasetTask, concat_tables
from relarena.models.rt.export import preprocessed_dir
from relarena.models.rt.model import DB_NAME


def precompute_dataset_task(
    dataset_name: str,
    task_name: str,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
) -> str:
    """Fill every RT tensor directory an inner+outer experiment reads."""
    cache = resolve_cache_config(cache_dir, on_miss="fill")
    if cache.directory is None:
        raise ValueError("RT warming needs --cache-dir or RELARENA_CACHE_DIR")
    source = RelBenchDatasetTask(dataset_name, task_name, download=download)

    inner = source.inner_split()
    outer = source.outer_split()
    # Every export a run reaches for. The selection arm trains on `train` and
    # validates on `val`; the reporting arm retrains on the train+val union and
    # then scores test beside it. A missing one of these is a `CacheMiss` at the
    # first fit of a configured run, not a slow run.
    #
    # Every export carrying a val or test split carries a train one too, because
    # rustler takes a task's normalizing statistics from the train split. The
    # two outer exports differ only by that test table, which carries no text --
    # so `export._embed` computes the text embeddings once and links the second.
    union = concat_tables(outer.train_table, outer.val_table)
    exports = [
        # the selection arm: train on train, validate on val
        ("inner", {"train": inner.train_table, "val": inner.eval_table}),
        # the reporting arm: refit on the union, then score test beside it
        ("outer", {"train": union}),
        ("outer", {"train": union, "test": outer.eval_table}),
    ]
    for phase, splits in exports:
        split = inner if phase == "inner" else outer
        preprocessed_dir(
            split.db_state,
            source.task,
            splits,
            cache=cache,
            identity=source.run_identity(phase),
            db_name=DB_NAME,
        )
    return f"{dataset_name}/{task_name}"


def main(argv: list[str] | None = None) -> int:
    """Warm one native RelBench task's RT tensor directories."""
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
