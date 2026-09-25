"""Build join keys for entity ids that survive a dtype difference.

Two tables can hold the same entity id under different dtypes. Joining them on
``astype(str)`` then compares ``"7"`` against ``"7.0"`` and matches nothing, and
a left join answers that with NaN instead of an error — so the joined column
silently disappears and the run still reports a score.

`to_stable_join_key` is what both sides of such a join should go through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_stable_join_key(entity_ids: pd.Series, column_name: str) -> pd.Series:
    """Turn entity ids into strings that match no matter their dtype.

    Whole numbers go through int64 first, so an int64 column and a float64
    column holding the same ids both come out as ``"7"``. Strings and
    categoricals are already stable and pass straight through.

    Args:
        entity_ids: The id column to convert. Missing values should already be
            rejected by the caller.
        column_name: Name of that column, used in the error message.

    Returns:
        One string key per id.

    Raises:
        ValueError: If a float id is not a whole number. No integer id column
            can produce one, so the two sides can never match, and failing here
            beats returning a column of NaN.
    """
    if pd.api.types.is_bool_dtype(entity_ids):
        return entity_ids.astype("int64").astype(str)
    if pd.api.types.is_float_dtype(entity_ids):
        numeric = entity_ids.to_numpy(dtype=float)
        if not np.array_equal(numeric, np.rint(numeric)):
            raise ValueError(
                f"entity ids in {column_name!r} must be whole numbers; a "
                "fractional id can never match the integer ids it is joined to"
            )
        return pd.Series(numeric.astype("int64"), index=entity_ids.index).astype(str)
    if pd.api.types.is_integer_dtype(entity_ids):
        return entity_ids.astype("int64").astype(str)
    return entity_ids.astype(str)


__all__ = ["to_stable_join_key"]
