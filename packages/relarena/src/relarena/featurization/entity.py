"""Entity-only featurization (the RelBench LightGBM recipe).

Left-join the label table to its entity table and use the entity's own columns plus
the anchor timestamp. There is **no** neighbor / foreign-key aggregation — the richer
multi-hop "deep feature synthesis" is a separate recipe (`dfs`).

Source: follows RelBench's LightGBM entity baseline, `examples/lightgbm_entity.py`
(https://github.com/snap-stanford/relbench) — the single label-table -> entity-table
join on the entity foreign key, entity columns only. Our version is torch-free
(pandas-based column typing) rather than using PyTorch Frame.
"""

from __future__ import annotations

import pandas as pd
from relbench.base import Database, EntityTask, Table

from relarena.featurization._columns import type_columns


def build_entity_features(
    task: EntityTask, db: Database, table: Table
) -> tuple[pd.DataFrame, list[str]]:
    """Build the entity-only feature table for a task's label `table`.

    Left-joins `table` (entity FK + anchor timestamp + target) to its entity
    table on `entity_col -> entity pkey`. Features are the entity's own columns
    plus the anchor timestamp; identifier (PK/FK) and target columns are excluded.

    Column handling:
      * datetime  -> float (nanoseconds since epoch; NaT -> NaN);
      * numeric   -> float;
      * bool / object / category -> kept as categorical (returned in the second
        element); the caller is responsible for a consistent encoding across splits;
      * list / array (embeddings) -> dropped (not usable by a plain GBDT here).

    Returns `(features_df, categorical_columns)`.
    """
    entity_table = db.table_dict[task.entity_table]
    entity_df = entity_table.df.astype(
        {entity_table.pkey_col: table.df[task.entity_col].dtype}
    )
    merged = table.df.merge(
        entity_df, how="left", left_on=task.entity_col, right_on=entity_table.pkey_col
    )

    drop = {task.target_col, task.entity_col, entity_table.pkey_col}
    drop |= set(table.fkey_col_to_pkey_table)
    drop |= set(entity_table.fkey_col_to_pkey_table)

    return type_columns(merged, drop)
