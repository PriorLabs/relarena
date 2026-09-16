"""Opt-in feature additions `tabpfn-rel` layers onto the DFS feature frame.

Each addition recovers signal that flat-DFS-into-one-TFM-call leaves behind, and is
appended *after* `build_dfs_features` (keyed off the label table's per-row cutoff
timestamp / entity key — information DFS drops):

  1. **Calendar** — cyclic sin/cos of the cutoff timestamp (stateless).
  2. **History lags** — per-entity lags of the target's own past values, read from a
     look-back pool built on the train labels (state: the pool).
  3. **Raw text** — the anchor table's free-text columns, passed through as raw
     strings for the TFM's estimator to embed (state: the column list).

`FeaturePipeline` assembles the additions a model config enables: `fit`
(on the full train labels) learns any state and returns the augmented train frame;
`transform` applies that state to a val/test frame. With every knob off it returns
the DFS frame unchanged, so the base `rdblearn` behavior is recovered exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table

# -- calendar ----------------------------------------------------------------

#: Calendar fields appended by `attach_calendar`, with their cyclic period.
_CALENDAR_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("month", "month", 12),
    ("dow", "dayofweek", 7),
    ("doy", "dayofyear", 365),
)


def attach_calendar(
    df: pd.DataFrame, raw_df: pd.DataFrame, time_col: str | None
) -> pd.DataFrame:
    """Append sin/cos calendar features of the cutoff timestamp to `df`.

    Adds `cutoff_{month,dow,doy}_{sin,cos}` from `raw_df[time_col]` (row-aligned
    with `df`) — the wraparound-safe cyclic complement to the raw integer
    year/month/day columns DFS already surfaces. A no-op when `time_col` is absent.
    """
    if time_col is None or time_col not in raw_df.columns:
        return df
    if len(df) != len(raw_df):
        raise ValueError(
            f"attach_calendar: row mismatch df={len(df)} raw={len(raw_df)}"
        )
    ts = pd.to_datetime(raw_df[time_col].to_numpy())
    out = df.copy()
    for name, attr, period in _CALENDAR_FIELDS:
        val = np.asarray(getattr(ts, attr), dtype="float64")
        out[f"cutoff_{name}_sin"] = np.sin(2 * np.pi * val / period)
        out[f"cutoff_{name}_cos"] = np.cos(2 * np.pi * val / period)
    return out


# -- history lags ------------------------------------------------------------


def build_history_pool(
    label_df: pd.DataFrame, *, entity_col: str, time_col: str, target_col: str
) -> pd.DataFrame:
    """The `[entity, cutoff, target]` rows used as the lag look-back pool.

    Built once at `fit` from the train label table and reused unchanged at
    `predict`, so a val/test anchor looks back into the same history the model was
    fit on.
    """
    return label_df[[entity_col, time_col, target_col]].copy()


def attach_history_lags(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    *,
    time_col: str,
    entity_col: str,
    target_col: str,
    n_lags: int,
) -> pd.DataFrame:
    """Append `target_lag{1..k}` and `target_lag{1..k}_age_days` per anchor.

    For each anchor row (from `raw_df`, row-aligned with `df`), looks back
    strictly *before* its cutoff within that entity's history (`history_df`) and
    attaches the `k` most recent `(target value, age in days)` pairs; `NaN` when
    the entity has fewer than `k` prior rows. The strict-less-than `merge_asof`
    matches fastdfs' cutoff join, so no post-cutoff label leaks in (an anchor never
    sees its own row). A no-op when `history_df` is `None` or `n_lags <= 0`.
    """
    if history_df is None or n_lags <= 0:
        return df
    if len(df) != len(raw_df):
        raise ValueError(
            f"attach_history_lags: row mismatch df={len(df)} raw={len(raw_df)}"
        )

    # Per-entity history sorted by time; shift(k-1) yields the k-th most recent
    # earlier row (the strict-LT snap below lands on a strictly-earlier row).
    hist = history_df[[entity_col, time_col, target_col]].copy()
    hist[time_col] = pd.to_datetime(hist[time_col])
    hist[entity_col] = hist[entity_col].astype(str)
    hist = hist.sort_values([entity_col, time_col], kind="mergesort").reset_index(
        drop=True
    )
    grouped = hist.groupby(entity_col, sort=False)
    for k in range(1, n_lags + 1):
        hist[f"_lag{k}_val"] = grouped[target_col].shift(k - 1)
        hist[f"_lag{k}_t"] = grouped[time_col].shift(k - 1)

    lag_cols = [f"_lag{k}_val" for k in range(1, n_lags + 1)] + [
        f"_lag{k}_t" for k in range(1, n_lags + 1)
    ]
    hist_for_merge = (
        hist[[entity_col, time_col, *lag_cols]]
        .sort_values(time_col, kind="mergesort")
        .reset_index(drop=True)
    )

    # Sentinel name avoids clobbering a task column literally called "index".
    orig_idx_col = "__rdbl_orig_idx__"
    anchors = raw_df[[entity_col, time_col]].reset_index(drop=True).copy()
    anchors[orig_idx_col] = np.arange(len(anchors), dtype=np.int64)
    anchors[time_col] = pd.to_datetime(anchors[time_col])
    anchors[entity_col] = anchors[entity_col].astype(str)
    anchors_sorted = anchors.sort_values(time_col, kind="mergesort")

    merged = pd.merge_asof(
        anchors_sorted,
        hist_for_merge,
        on=time_col,
        by=entity_col,
        allow_exact_matches=False,  # strict-LT: never snap to the anchor's own row
        direction="backward",
    )

    # Undo the time-sort to restore the original (df-aligned) row order.
    order = merged[orig_idx_col].to_numpy()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = df.reset_index(drop=True).copy()
    for k in range(1, n_lags + 1):
        out[f"target_lag{k}"] = merged[f"_lag{k}_val"].to_numpy()[inv]
        age_days = (merged[time_col] - merged[f"_lag{k}_t"]) / pd.Timedelta("1D")
        out[f"target_lag{k}_age_days"] = age_days.to_numpy()[inv]
    return out


# -- text --------------------------------------------------------------------


def anchor_text_columns(db: Database, task: EntityTask) -> list[str]:
    """The free-text columns on the task's entity (anchor) table.

    Object/string columns minus the anchor's keys and time column. We select by
    pandas dtype (relarena has no semantic-type metadata) and leave it to the
    estimator to treat genuine low-cardinality columns as categoricals rather
    than text.
    """
    anchor = db.table_dict[task.entity_table]
    adf = anchor.df
    exclude = {anchor.pkey_col, anchor.time_col, *anchor.fkey_col_to_pkey_table}
    return [
        c
        for c in adf.columns
        if c not in exclude
        and (adf[c].dtype == object or pd.api.types.is_string_dtype(adf[c]))
    ]


def _lookup_anchor_text(
    db: Database, task: EntityTask, split_df: pd.DataFrame, text_cols: list[str]
) -> pd.DataFrame:
    """Fetch the anchor table's `text_cols` for each row of `split_df` by key.

    Most RelBench task tables hold only `(entity_key, timestamp, label)`; the text
    lives on the anchor table, looked up by the entity key. Returns a frame
    row-aligned with `split_df` (left join, so missing keys give NaN).
    """
    anchor = db.table_dict[task.entity_table]
    adf = anchor.df
    pk = anchor.pkey_col
    available = [c for c in text_cols if c in adf.columns]
    if not available:
        return pd.DataFrame(index=range(len(split_df)))

    lookup = adf[[pk, *available]].drop_duplicates(subset=[pk]).copy()
    # Coerce to plain str — some anchor tables store list/ndarray cells that break
    # numeric coercion downstream; str() still embeds meaningfully.
    for c in available:
        lookup[c] = lookup[c].astype(str)
    lookup[pk] = lookup[pk].astype(str)

    ent = task.entity_col
    raw = split_df.reset_index(drop=True)[[ent]].copy()
    raw[ent] = raw[ent].astype(str)
    joined = raw.merge(lookup, left_on=ent, right_on=pk, how="left")
    return joined[available]


def attach_text(df: pd.DataFrame, text: pd.DataFrame | None) -> pd.DataFrame:
    """Concatenate the (row-aligned) text columns onto `df`."""
    if text is None or text.empty:
        return df
    if len(df) != len(text):
        raise ValueError(f"attach_text: row mismatch df={len(df)} text={len(text)}")
    return pd.concat([df.reset_index(drop=True), text.reset_index(drop=True)], axis=1)


class RawTextFeaturizer:
    """Anchor-text columns passed through as raw strings.

    The TFM's estimator handles them (see
    `relarena.models._shared.tfm.tfm`). Columns are
    suffixed `__raw_text` so a low-cardinality anchor column DFS kept as a
    categorical cannot collide. Stateless beyond the column list and uncached
    (the lookup is a cheap merge).
    """

    def __init__(self) -> None:
        """Start unfitted; the column list is detected at fit."""
        self._cols: list[str] | None = None

    def fit(
        self, db: Database, task: EntityTask, train_table: Table
    ) -> pd.DataFrame | None:
        """Detect the anchor text columns and return the train text frame."""
        self._cols = anchor_text_columns(db, task)
        return self.transform(db, task, train_table)

    def transform(
        self, db: Database, task: EntityTask, table: Table
    ) -> pd.DataFrame | None:
        """Return the split's raw text frame (no-op without text columns)."""
        if not self._cols:
            return None
        raw = _lookup_anchor_text(db, task, table.df, self._cols)
        return raw.rename(columns={c: f"{c}__raw_text" for c in raw.columns})


# -- the assembled pipeline --------------------------------------------------


class FeaturePipeline:
    """The opt-in feature additions a `tabpfn-rel` config enables (see module doc).

    Additions are applied in a fixed order — calendar, history lags, text — so the
    train and val/test frames carry an identical column schema. Stateless with every
    knob off.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Read which feature additions the model `config` enables."""
        self.calendar = bool(config.get("with_calendar_features"))
        self.n_lags = (
            int(config.get("n_lags", 5)) if config.get("with_history_features") else 0
        )
        self.text = RawTextFeaturizer() if config.get("with_text") else None
        self._history_pool: pd.DataFrame | None = None

    @property
    def enabled(self) -> bool:
        """Whether any addition is active."""
        return self.calendar or self.n_lags > 0 or self.text is not None

    def fit_transform(
        self, df: pd.DataFrame, task: EntityTask, train_table: Table, db: Database
    ) -> pd.DataFrame:
        """Learn state on the train labels and return the augmented train frame."""
        if self.calendar:
            df = attach_calendar(df, train_table.df, task.time_col)
        if self.n_lags > 0:
            self._history_pool = build_history_pool(
                train_table.df,
                entity_col=task.entity_col,
                time_col=task.time_col,
                target_col=task.target_col,
            )
            df = self._attach_lags(df, train_table.df, task)
        if self.text is not None:
            df = attach_text(df, self.text.fit(db, task, train_table))
        return df

    def transform(
        self, df: pd.DataFrame, task: EntityTask, table: Table, db: Database
    ) -> pd.DataFrame:
        """Apply the learned state to a val/test frame."""
        if self.calendar:
            df = attach_calendar(df, table.df, task.time_col)
        if self.n_lags > 0:
            df = self._attach_lags(df, table.df, task)
        if self.text is not None:
            df = attach_text(df, self.text.transform(db, task, table))
        return df

    def _attach_lags(
        self, df: pd.DataFrame, raw_df: pd.DataFrame, task: EntityTask
    ) -> pd.DataFrame:
        return attach_history_lags(
            df,
            raw_df,
            self._history_pool,
            time_col=task.time_col,
            entity_col=task.entity_col,
            target_col=task.target_col,
            n_lags=self.n_lags,
        )
