"""Production inference: predict label-less rows at an arbitrary anchor time.

Unlike the train/val/test tables, a production prediction is made at an anchor
whose forward window `(t, t + timedelta]` lies in the future — so no label can
be (or needs to be) computed. `make_prediction_table` builds the label-less
seed of `(entity, anchor)` rows directly (skipping the label-generating SQL),
and `predict_at` censors the database to the anchor and runs `model.predict`
on it — the same call the evaluation path already makes on the masked test table.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table

from relarena.dataset import _copy_timeless_tables

if TYPE_CHECKING:
    from relarena.model import RelArenaModel

#: Entity universe selector: `"all"` (every entity existing at the anchor) or an
#: explicit collection of entity ids.
EntitySelector = Union[str, Sequence[int]]


def make_prediction_table(
    db: Database,
    task: EntityTask,
    at_timestamp: pd.Timestamp,
    entities: EntitySelector = "all",
) -> Table:
    """Build a label-less seed table of `(entity, at_timestamp)` rows to score.

    The seed has no target column — `model.predict` consumes only the entity and
    time columns. `entities` is either `"all"` (every entity that exists at the
    anchor, i.e. whose row in the entity table is dated `<= at_timestamp` when the
    entity table is temporal) or an explicit collection of entity ids.
    """
    at_timestamp = pd.Timestamp(at_timestamp)
    entity_tbl = db.table_dict[task.entity_table]

    if isinstance(entities, str):
        if entities != "all":
            raise ValueError(
                f"Unknown entity selector {entities!r}; use 'all' or an id list."
            )
        ids = entity_tbl.df[entity_tbl.pkey_col]
        if entity_tbl.time_col is not None:
            ids = ids[entity_tbl.df[entity_tbl.time_col] <= at_timestamp]
        ids = ids.to_numpy()
    else:
        if not hasattr(entities, "__iter__"):
            raise ValueError(
                f"entities must be 'all' or a list of ids; got scalar {entities!r} "
                "- wrap a single id in a list, e.g. [5]."
            )
        ids = np.asarray(list(entities))
        # Drop explicit ids absent from the entity table (a temporal one also
        # restricts to entities existing by the anchor); warn about any dropped.
        present = entity_tbl.df[entity_tbl.pkey_col]
        if entity_tbl.time_col is not None:
            present = present[entity_tbl.df[entity_tbl.time_col] <= at_timestamp]
        keep = np.isin(ids, present.to_numpy())
        if not keep.all():
            warnings.warn(
                f"Dropped {int((~keep).sum())} requested entity id(s) absent "
                f"from the database: {ids[~keep].tolist()}",
                stacklevel=2,
            )
        ids = ids[keep]

    df = pd.DataFrame({task.time_col: at_timestamp, task.entity_col: ids})
    return Table(
        df=df,
        fkey_col_to_pkey_table={task.entity_col: task.entity_table},
        pkey_col=None,
        time_col=task.time_col,
    )


def predict_at(
    model: RelArenaModel,
    task: EntityTask,
    db: Database,
    at_timestamp: pd.Timestamp,
    entities: EntitySelector = "all",
) -> pd.DataFrame:
    """Predict the task target for label-less rows at `at_timestamp`.

    Censors the database to `<= at_timestamp` (so the model sees only data
    available at the anchor — no leakage), builds the label-less seed, and runs
    `model.predict`. Returns a frame of entity ids, the anchor time, and the
    predicted target (one row per scored entity).
    """
    at_timestamp = pd.Timestamp(at_timestamp)
    seed = make_prediction_table(db, task, at_timestamp, entities)
    db_now = db.upto(at_timestamp)
    # Scrub FKs left dangling by censoring - timeless child rows can still point at
    # entities upto just removed - mirroring the split path. Copy the shared timeless
    # tables first so the in-place scrub can't corrupt the caller's db.
    _copy_timeless_tables(db_now)
    task.dataset.validate_and_correct_db(db_now)
    preds = model.predict(task, db_now, seed)
    return pd.DataFrame(
        {
            task.entity_col: seed.df[task.entity_col].to_numpy(),
            task.time_col: at_timestamp,
            f"{task.target_col}_pred": preds,
        }
    )
