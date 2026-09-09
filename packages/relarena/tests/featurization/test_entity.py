"""Tests for entity-only featurization (no data download — stand-in tables)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from relbench.base import Table, TaskType

from relarena.featurization import build_entity_features


def _setup() -> tuple[SimpleNamespace, SimpleNamespace, Table]:
    entity = Table(
        df=pd.DataFrame(
            {
                "uid": [1, 2, 3],
                "country": ["US", "DE", "US"],  # categorical
                "age": [20, 30, 40],  # numeric
                "signup": pd.to_datetime(
                    ["2020-01-01", "2020-06-01", "2021-01-01"]
                ),  # datetime
                "emb": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # list -> dropped
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col="uid",
        time_col=None,
    )
    label = Table(
        df=pd.DataFrame(
            {
                "uid": [1, 2, 1],
                "date": pd.to_datetime(["2021-02-01", "2021-02-01", "2021-03-01"]),
                "y": [0, 1, 1],
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="date",
    )
    db = SimpleNamespace(table_dict={"users": entity})
    task = SimpleNamespace(
        entity_col="uid",
        entity_table="users",
        target_col="y",
        time_col="date",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    return task, db, label


def test_drops_identifiers_target_and_embeddings() -> None:
    task, db, label = _setup()
    feats, categorical = build_entity_features(task, db, label)
    assert len(feats) == 3
    assert "uid" not in feats.columns  # entity_col / pkey / fkey
    assert "y" not in feats.columns  # target
    assert "emb" not in feats.columns  # list/embedding column dropped


def test_column_typing_and_categoricals() -> None:
    task, db, label = _setup()
    feats, categorical = build_entity_features(task, db, label)
    assert categorical == ["country"]
    # numeric + datetime are floats; datetimes (signup, anchor date) included
    assert {"age", "signup", "date"} <= set(feats.columns)
    assert feats["age"].dtype == float
    assert feats["signup"].dtype == float
    assert feats["date"].dtype == float


def test_join_correctness() -> None:
    task, db, label = _setup()
    feats, _ = build_entity_features(task, db, label)
    # row 0: uid=1 -> US, age 20; row 1: uid=2 -> DE, age 30
    assert feats.iloc[0]["country"] == "US" and feats.iloc[0]["age"] == 20.0
    assert feats.iloc[1]["country"] == "DE" and feats.iloc[1]["age"] == 30.0
