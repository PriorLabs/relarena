"""Anchor-text lookup adapted and modified from TabPFN-Rel.

Copyright 2026 PriorLabs GmbH. Licensed under Apache-2.0; see
``APACHE-2.0.txt``. Synthefy added prose filtering and validated joins for
Nori's text pipeline. The source is ``relarena.models.tabpfn_rel.features`` at
RelArena commit ``efac8f26435ff4cb682b2ca19329fb391a2bfa58``.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd
from relbench.base import Database, EntityTask

from .keys import to_stable_join_key

RAW_TEXT_SUFFIX = "__raw_text"
TEXT_SAMPLE_ROWS = 10_000
MIN_MEAN_TEXT_LENGTH = 20
IDENTIFIER_TOKENS = frozenset({"code", "id", "name", "postal", "ref", "url", "zip"})
#: What `attach_anchor_text` turns a source null into when it stringifies a
#: column: a float NaN, a Python `None` and a pandas `NA` respectively. A column
#: made only of these is empty, although `dropna` cannot see it.
STRINGIFIED_NULLS: Final = frozenset({str(float("nan")), str(None), str(pd.NA)})


def _is_text(column: str, values: pd.Series) -> bool:
    """Separate prose-like strings from identifiers and short categoricals."""
    tokens = set(re.findall(r"[a-z0-9]+", column.lower()))
    if tokens & IDENTIFIER_TOKENS or column.lower().endswith("url"):
        return False
    sample = values.dropna().head(TEXT_SAMPLE_ROWS).astype(str)
    return not sample.empty and sample.str.len().mean() >= MIN_MEAN_TEXT_LENGTH


def anchor_text_columns(db: Database, task: EntityTask) -> list[str]:
    """Return prose-like, non-key strings from the task's anchor table."""
    anchor = db.table_dict[task.entity_table]
    excluded = {anchor.pkey_col, anchor.time_col, *anchor.fkey_col_to_pkey_table}
    return [
        column
        for column in anchor.df.columns
        if column not in excluded
        and (
            anchor.df[column].dtype == object
            or pd.api.types.is_string_dtype(anchor.df[column])
        )
        and _is_text(column, anchor.df[column])
    ]


def attach_anchor_text(
    features: pd.DataFrame,
    db: Database,
    task: EntityTask,
    split: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Join the anchor entity's text to a row-aligned feature frame."""
    if not columns:
        return features, []
    if len(features) != len(split):
        raise ValueError(
            f"text row mismatch: features={len(features)}, split={len(split)}"
        )

    anchor = db.table_dict[task.entity_table]
    primary_key = anchor.pkey_col
    available = [column for column in columns if column in anchor.df.columns]
    lookup = anchor.df[[primary_key, *available]].drop_duplicates(primary_key).copy()
    lookup[primary_key] = to_stable_join_key(lookup[primary_key], primary_key)
    for column in available:
        lookup[column] = lookup[column].astype(str)

    entity_key = task.entity_col
    keys = split.reset_index(drop=True)[[entity_key]].copy()
    keys[entity_key] = to_stable_join_key(keys[entity_key], entity_key)
    joined = keys.merge(
        lookup,
        left_on=entity_key,
        right_on=primary_key,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    text = joined[available].copy()
    names = [f"{column}{RAW_TEXT_SUFFIX}" for column in available]
    text.columns = names
    return pd.concat([features.reset_index(drop=True), text], axis=1), names
