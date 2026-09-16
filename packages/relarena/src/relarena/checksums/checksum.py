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

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table

from relarena.dataset import RelBenchDatasetTask, drop_noncanonical_task_columns

#: Recorded baseline, shipped as package data beside this module.
CHECKSUMS_PATH = Path(__file__).with_name("relbench_v1_checksums.json")

#: Byte width -> unsigned view dtype, for reinterpreting any fixed-width column.
_BYTES_TO_UINT: dict[int, type] = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
}


def array_checksum(arr: np.ndarray) -> np.uint64:
    """A `uint64` checksum of a numpy array (ported from `benchmarking`).

    XOR-reduces the values, bit-rotating within blocks of `num_bits` by the
    within-block index (so position matters and 0/1-heavy columns don't collapse)
    and normalizing endianness. Requires a fixed-width dtype — reduce
    object/string/list columns via `_column_codes` first.
    """
    itemsize = np.dtype(arr.dtype).itemsize
    uint_dtype = _BYTES_TO_UINT.get(itemsize)
    if uint_dtype is None:
        raise ValueError(f"Unsupported dtype {arr.dtype}")
    num_bits = itemsize * 8
    arr = arr.view(uint_dtype).flatten()
    pad = (num_bits - arr.size % num_bits) % num_bits
    blocks = np.pad(arr, (0, pad), mode="constant").reshape(-1, num_bits)
    if sys.byteorder == "big":
        blocks = blocks.byteswap(inplace=False)
    left = np.arange(num_bits, dtype=np.uint8)
    right = np.arange(num_bits, 0, -1, dtype=np.uint8)
    rotated = (blocks << left) | (blocks >> right)
    return np.bitwise_xor.reduce(rotated.flatten(), dtype=np.uint64)


def _hash64(text: str) -> np.uint64:
    """Deterministic `uint64` from a string (first 8 bytes of its SHA256)."""
    return np.uint64(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def _to_hashable(v: object) -> object:
    """Turn array/list cells into hashable tuples; leave scalars untouched."""
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    if isinstance(v, list):
        return tuple(v)
    return v


def _column_codes(s: pd.Series) -> np.ndarray:
    """Reduce a column to one `uint64` per row, handling every RelBench dtype.

    Fixed-width numeric/bool/datetime arrays are reinterpreted directly; strings,
    `list`-valued columns, and nullable dtypes (object under `to_numpy()`) go
    through pandas' C-level row hasher, with a per-row fallback for the unhashable
    (array/list) cells.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.to_numpy(dtype="datetime64[ns]").view(np.int64)
    a = s.to_numpy()
    if a.dtype.kind == "b":  # real (non-nullable) bool
        return a.astype(np.uint8)
    if a.dtype.kind in "iuf":  # real fixed-width numeric
        return a if a.dtype.itemsize in _BYTES_TO_UINT else a.astype(np.float64)
    col = pd.Series(a)
    try:
        return pd.util.hash_pandas_object(col, index=False).to_numpy()
    except TypeError:  # unhashable cells (e.g. list<...> columns)
        return pd.util.hash_pandas_object(col.map(_to_hashable), index=False).to_numpy()


def table_checksum(table: Table) -> np.uint64:
    """Integer content checksum of a `Table` (data + relational schema)."""
    cs = np.uint64(0)
    for i, col in enumerate(sorted(table.df.columns, key=str)):
        # Roll each column by its index so identical value-arrays in different
        # columns don't cancel under XOR; fold in the column name + dtype.
        cs ^= array_checksum(np.roll(_column_codes(table.df[col]), i))
        cs ^= _hash64(f"{col}:{table.df[col].dtype}")
    cs ^= _hash64(
        f"fkey={sorted(table.fkey_col_to_pkey_table.items())}"
        f"|pkey={table.pkey_col}|time={table.time_col}"
    )
    return cs


def database_checksum(db: Database) -> np.uint64:
    """Integer content checksum of a `Database` (each table, hashed by name).

    Name and content are hashed *jointly* so swapping which table sits under which
    name changes the result.
    """
    cs = np.uint64(0)
    for name in sorted(db.table_dict):
        cs ^= _hash64(f"{name}|{int(table_checksum(db.table_dict[name]))}")
    return cs


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
