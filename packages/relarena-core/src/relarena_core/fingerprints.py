"""Content fingerprints for relational tables and databases."""

import hashlib
import sys

import numpy as np
import pandas as pd
from relbench.base import Database, Table

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
