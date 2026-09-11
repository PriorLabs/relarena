"""Integer content checksums for RelBench tables, databases, and task splits.

A fast `uint64` XOR checksum over the fully materialized data (ported from
`benchmarking.datasets`), recorded per `(dataset, task)` in
the JSON beside this module. RelBench tables mix dtypes — including
`list`-valued columns (e.g. `product.category`) and pandas *nullable* dtypes
that break a naive `pd.util.hash_pandas_object` — so `_column_codes`
first reduces any column to one `uint64` per row.

The baseline pins the **model-facing split objects** (censored, column-dropped
databases + label tables of `inner_split()`/`outer_split()`, plus the hidden
test labels), so it must be re-recorded whenever upstream data *or* our own
load-time processing (`drop_noncanonical_columns`) changes.

These are pure helpers. The (slow, download-bound) recorder/checker CLI that
drives them lives in `workflows/record_checksums.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from relbench.base import EntityTask, Table

from relarena.core.fingerprints import array_checksum, database_checksum, table_checksum
from relarena.dataset import RelBenchDatasetTask, drop_noncanonical_task_columns

#: Recorded baseline, shipped as package data beside this module.
CHECKSUMS_PATH = Path(__file__).with_name("relbench_v1_checksums.json")


def _db_checksums(source: RelBenchDatasetTask) -> dict[str, int]:
    """Checksums of the censored inner/outer databases (dataset-level, slow)."""
    return {
        "inner_db": int(database_checksum(source.inner_split().db_state)),
        "outer_db": int(database_checksum(source.outer_split().db_state)),
    }


def _canonical_label_order(table: Table, task: EntityTask) -> Table:
    """Return `table` sorted by `(time_col, entity_col)` for an order-stable checksum.

    Native RelBench tasks build label tables with unordered DuckDB queries, whose
    row order is nondeterministic across recomputes, and `table_checksum` is
    row-order-sensitive. The `(time_col, entity_col)` pair is unique per label row,
    so sorting on it gives a deterministic total order.
    """
    return Table(
        df=table.df.sort_values([task.time_col, task.entity_col]).reset_index(
            drop=True
        ),
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def _label_checksums(source: RelBenchDatasetTask) -> dict[str, int]:
    """Checksums of the split label tables, plus the hidden test labels.

    Each table is put in canonical `(time_col, entity_col)` order first, so the
    checksum is stable across native RelBench's nondeterministic label row order
    (see `_canonical_label_order`).
    """
    inner, outer = source.inner_split(), source.outer_split()
    task = source.task

    def label_cs(table: Table) -> int:
        return int(table_checksum(_canonical_label_order(table, task)))

    return {
        "inner_train": label_cs(inner.train_table),
        "inner_eval": label_cs(inner.eval_table),
        # Outer train and val are fingerprinted independently — the split exposes them
        # separately, and how a model combines them is the per-model final-fit regime.
        "outer_train": label_cs(outer.train_table),
        "outer_val": label_cs(outer.val_table),
        "outer_eval": label_cs(outer.eval_table),
        "test_labels": label_cs(
            drop_noncanonical_task_columns(
                task,
                task.get_table("test", mask_input_cols=False),
                source.dataset_name,
            )
        ),
    }


def split_checksums(
    dataset_name: str, task_name: str, *, download: bool = True
) -> dict[str, int]:
    """Full checksums for one task's inner/outer splits, as a model sees them."""
    source = RelBenchDatasetTask(dataset_name, task_name, download=download)
    return {**_db_checksums(source), **_label_checksums(source)}


def _iter_checksums(
    specs: list[tuple[str, str]], *, download: bool = True
) -> Iterator[tuple[str, dict[str, int]]]:
    """Yield `(key, checksums)` per `(dataset, task)`, hashing each DB once.

    The `inner_db`/`outer_db` checksums depend only on the dataset, so they
    are cached and reused across that dataset's tasks (the expensive part).
    """
    db_cache: dict[str, dict[str, int]] = {}
    for dataset_name, task_name in specs:
        source = RelBenchDatasetTask(dataset_name, task_name, download=download)
        if dataset_name not in db_cache:
            db_cache[dataset_name] = _db_checksums(source)
        yield (
            f"{dataset_name}/{task_name}",
            {**db_cache[dataset_name], **_label_checksums(source)},
        )


def record_checksums(
    specs: list[tuple[str, str]],
    output_path: Path = CHECKSUMS_PATH,
    *,
    download: bool = True,
) -> dict[str, dict[str, int]]:
    """Compute and persist full-split checksums for `(dataset, task)` `specs`.

    Merges into any existing baseline, writing after every task so a long run's
    progress survives an interruption.
    """
    try:
        baseline = json.loads(output_path.read_text())
    except FileNotFoundError:
        baseline = {}
    for key, checksums in _iter_checksums(specs, download=download):
        print(f"recording {key} ...", flush=True)
        baseline[key] = checksums
        output_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(baseline)} task checksums to {output_path}")
    return baseline


def check_checksums(
    specs: list[tuple[str, str]],
    output_path: Path = CHECKSUMS_PATH,
    *,
    download: bool = True,
) -> dict[str, dict[str, tuple[int | None, int | None]]]:
    """Recompute and compare against the recorded baseline **without writing**.

    Returns mismatches as `{key: {split: (recorded, computed)}}` (empty when all
    match); a missing task or one-sided split is reported with `None` on the
    absent side.
    """
    baseline = json.loads(output_path.read_text())
    mismatches: dict[str, dict[str, tuple[int | None, int | None]]] = {}
    for key, computed in _iter_checksums(specs, download=download):
        recorded = baseline.get(key, {})
        diff = {
            split: (recorded.get(split), computed.get(split))
            for split in recorded.keys() | computed.keys()
            if recorded.get(split) != computed.get(split)
        }
        status = "MISMATCH" if diff else "ok"
        if key not in baseline:
            status = "MISSING (not in baseline)"
        print(f"{status:8s} {key}", flush=True)
        if diff:
            mismatches[key] = diff
    return mismatches


__all__ = [
    "array_checksum",
    "table_checksum",
    "database_checksum",
    "split_checksums",
    "record_checksums",
    "check_checksums",
    "CHECKSUMS_PATH",
]
