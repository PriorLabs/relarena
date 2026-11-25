"""Select and preserve target semantics using only supplied fitting labels."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from relbench.base import TaskType

EXACT = 2**53


def integral(values: pa.Array | pa.ChunkedArray) -> bool:
    """Check exact integer ranks before any potentially lossy float conversion."""
    if pa.types.is_dictionary(values.type):
        values = pc.dictionary_decode(values)
    if not len(values) or values.null_count:
        return False
    datatype = values.type
    if pa.types.is_integer(datatype):
        upper = pa.scalar(
            EXACT,
            type=pa.uint64() if pa.types.is_unsigned_integer(datatype) else pa.int64(),
        )
        invalid = pc.greater(values, upper)
        if pa.types.is_signed_integer(datatype):
            invalid = pc.or_(
                invalid, pc.less(values, pa.scalar(-EXACT, type=pa.int64()))
            )
        return not pc.any(invalid).as_py()
    if pa.types.is_floating(datatype):
        valid = pc.and_(pc.is_finite(values), pc.equal(values, pc.floor(values)))
        valid = pc.and_(
            valid,
            pc.and_(pc.greater_equal(values, -EXACT), pc.less_equal(values, EXACT)),
        )
        return bool(pc.all(valid).as_py())
    return False


def kind(task_type: TaskType, values: pa.Array | pa.ChunkedArray) -> str:
    """Infer a regression representation without reading another data split."""
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "boolean"
    if task_type == TaskType.REGRESSION:
        return "ordinal" if integral(values) else "number"
    raise ValueError(
        f"relflow has no target representation for task type {task_type!r}."
    )


def encode(
    values: pa.Array | pa.ChunkedArray, selected: str
) -> pa.Array | pa.ChunkedArray:
    """Apply a frozen target kind; later labelled inputs cannot redefine it."""
    if selected == "ordinal":
        if not integral(values):
            raise ValueError(
                "Frozen relflow ordinal targets require finite, non-null integer "
                "labels within [-2**53, 2**53]; fractional or out-of-range values "
                "cannot be cast."
            )
        return pc.cast(values, pa.int64(), safe=True)
    if selected == "number":
        return pc.cast(values, pa.float64())
    if selected == "boolean":
        if not pa.types.is_boolean(values.type):
            invalid = pc.and_(pc.not_equal(values, 0), pc.not_equal(values, 1))
            if pc.any(invalid).as_py():
                raise ValueError("Binary training labels must be 0 or 1.")
        return pc.cast(values, pa.bool_())
    raise ValueError(f"Unsupported frozen relflow target kind {selected!r}.")
