"""Shared column typing for featurization recipes.

Given a raw feature frame (after a join or DFS) and a set of columns to drop
(identifiers, target, timestamps), split the rest into numeric (float) and
categorical columns:

  * datetime  -> float (nanoseconds since epoch; NaT -> NaN);
  * bool      -> categorical (object);
  * numeric   -> float;
  * object / category -> categorical (object);
  * list / array (embeddings, multicategorical) -> dropped.

Returns `(features_df, categorical_columns)`; the caller freezes a consistent
categorical encoding across splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_listlike_column(series: pd.Series) -> bool:
    """Whether a column holds list/array values (embeddings, multicategorical)."""
    for v in series:
        if v is None or (np.ndim(v) == 0 and pd.isna(v)):
            continue
        return isinstance(v, (list, tuple, np.ndarray))
    return False


def type_columns(df: pd.DataFrame, drop: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Split `df` into numeric + categorical features, excluding `drop` columns."""
    numeric: dict[str, pd.Series] = {}
    categorical: dict[str, pd.Series] = {}
    for col in df.columns:
        if col in drop:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            ns = s.astype("int64").astype("float64")
            ns[s.isna()] = np.nan
            numeric[col] = ns
        elif pd.api.types.is_bool_dtype(s):
            categorical[col] = s.astype("object")
        elif pd.api.types.is_numeric_dtype(s):
            numeric[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif is_listlike_column(s):
            continue
        else:
            categorical[col] = s.astype("object")

    feats = pd.DataFrame({**numeric, **categorical})
    return feats, list(categorical)
