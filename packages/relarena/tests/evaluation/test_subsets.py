"""Tests for the task subsets behind custom leaderboards."""

from __future__ import annotations

import pandas as pd
import pytest

from relarena.evaluation import apply_subset

_RESULTS = pd.DataFrame(
    [
        {"model": "lightgbm", "dataset": "rel-f1", "task_type": "REGRESSION"},
        {
            "model": "lightgbm",
            "dataset": "rel-hm",
            "task_type": "BINARY_CLASSIFICATION",
        },
    ]
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("all", {"REGRESSION", "BINARY_CLASSIFICATION"}),
        ("regression", {"REGRESSION"}),
        ("classification", {"BINARY_CLASSIFICATION"}),
    ],
)
def test__apply_subset__name__keeps_that_task_type(
    name: str, expected: set[str]
) -> None:
    assert set(apply_subset(_RESULTS, name)["task_type"]) == expected


def test__apply_subset__task_type_names__partition_the_frame() -> None:
    # Guard: the two task-type subsets must not overlap or leave a task behind,
    # or a board built per task type would double-count or silently skip one.
    classification = apply_subset(_RESULTS, "classification")
    regression = apply_subset(_RESULTS, "regression")
    assert len(classification) + len(regression) == len(_RESULTS)


def test__apply_subset__own_mask__filters_by_it() -> None:
    only_f1 = apply_subset(_RESULTS, lambda d: d["dataset"] == "rel-f1")
    assert set(only_f1["dataset"]) == {"rel-f1"}


def test__apply_subset__unknown_name__raises_listing_the_known_ones() -> None:
    with pytest.raises(ValueError, match="Unknown task subset 'tiny'"):
        apply_subset(_RESULTS, "tiny")


def test__apply_subset__all__works_without_task_metadata() -> None:
    # The default must not require any column beyond what it is handed.
    assert len(apply_subset(pd.DataFrame({"model": ["lightgbm"]}), "all")) == 1
