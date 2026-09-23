"""Tests for row-aligned RelArena text features."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from relarena.models.nori_rel.text import anchor_text_columns, attach_anchor_text


def _database() -> Any:
    anchor = SimpleNamespace(
        df=pd.DataFrame(
            {
                "item_id": [1, 2],
                "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "owner_id": [10, 20],
                "title": [
                    "first sufficiently long title",
                    "second sufficiently long title",
                ],
                "description": [
                    ["one long description", "two long description"],
                    None,
                ],
                "facility_name": [
                    "first long facility name",
                    "second long facility name",
                ],
                "website_url": [
                    "https://example.com/one",
                    "https://example.com/two",
                ],
                "price": [3.0, 4.0],
            }
        ),
        pkey_col="item_id",
        time_col="created_at",
        fkey_col_to_pkey_table={"owner_id": "users"},
    )
    return SimpleNamespace(table_dict={"items": anchor})


def test_anchor_text_columns_exclude_keys_and_short_strings() -> None:
    task = SimpleNamespace(entity_table="items")

    assert anchor_text_columns(_database(), task) == ["title", "description"]


def test_attach_anchor_text_preserves_split_order() -> None:
    task = SimpleNamespace(entity_table="items", entity_col="item_id")
    split = pd.DataFrame({"item_id": [2, 9, 1]})
    features = pd.DataFrame({"value": [20.0, 90.0, 10.0]})

    output, columns = attach_anchor_text(
        features,
        _database(),
        task,
        split,
        ["title", "description"],
    )

    assert columns == ["title__raw_text", "description__raw_text"]
    assert output["title__raw_text"].iloc[[0, 2]].tolist() == [
        "second sufficiently long title",
        "first sufficiently long title",
    ]
    assert output["description__raw_text"].iloc[2] == (
        "['one long description', 'two long description']"
    )
    assert np.isnan(output["title__raw_text"].iloc[1])


def test_float_entity_keys_still_match_integer_anchor_keys() -> None:
    db = _database()
    task = SimpleNamespace(entity_table="items", entity_col="item_id")
    # The anchor's item_id is int64. One missing value anywhere upstream upcasts
    # the task's own entity column to float64, which used to render the same id
    # as "1.0" against the anchor's "1" and silently return a column of NaN text.
    split = pd.DataFrame({"item_id": np.array([1.0, 2.0], dtype="float64")})
    features = pd.DataFrame({"row": [0, 1]})

    output, names = attach_anchor_text(features, db, task, split, ["title"])

    assert names == ["title__raw_text"]
    assert output["title__raw_text"].tolist() == [
        "first sufficiently long title",
        "second sufficiently long title",
    ]


def test_fractional_entity_keys_fail_closed() -> None:
    db = _database()
    task = SimpleNamespace(entity_table="items", entity_col="item_id")
    split = pd.DataFrame({"item_id": [1.5, 2.5]})
    features = pd.DataFrame({"row": [0, 1]})

    with pytest.raises(ValueError, match="must be whole numbers"):
        attach_anchor_text(features, db, task, split, ["title"])


def test_attach_anchor_text_obeys_strict_entity_cutoff() -> None:
    task = SimpleNamespace(
        entity_table="items",
        entity_col="item_id",
        time_col="cutoff",
    )
    split = pd.DataFrame(
        {
            "item_id": [1, 2, 2],
            "cutoff": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-01"]),
        }
    )

    output, _ = attach_anchor_text(
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}),
        _database(),
        task,
        split,
        ["title"],
        strict_cutoff=True,
    )

    assert output["title__raw_text"].iloc[0] == "first sufficiently long title"
    assert output["title__raw_text"].iloc[1:].isna().all()
