"""Tests for task discovery.

These read RelBench's task registry (class attributes only) and must NOT trigger
any dataset download.
"""

from __future__ import annotations

from relbench.base import TaskType

from relarena.tasks import ENTITY_TASK_TYPES, RELBENCH_V1_DATASETS, list_entity_tasks


def test_rel_f1_entity_tasks_only() -> None:
    specs = list_entity_tasks(["rel-f1"])
    names = {s.task for s in specs}

    # entity classification + regression are included
    assert {"driver-dnf", "driver-top3", "driver-position"} <= names
    # link prediction is excluded
    assert "driver-circuit-compete" not in names
    # autocomplete tasks are excluded (they subclass EntityTask!)
    assert "results-position" not in names
    assert "qualifying-position" not in names
    # every discovered task is an in-scope entity type
    assert all(s.task_type in ENTITY_TASK_TYPES for s in specs)


def test_multiclass_excluded() -> None:
    # rel-arxiv's author-category is a multiclass entity task; it must NOT be
    # discovered, and no discovered task may be multiclass.
    specs = list_entity_tasks(["rel-arxiv"])
    assert not any(s.task == "author-category" for s in specs)
    assert all(s.task_type != TaskType.MULTICLASS_CLASSIFICATION for s in specs)


def test_v1_default_scan_has_expected_count() -> None:
    specs = list_entity_tasks(list(RELBENCH_V1_DATASETS))
    # 21 entity classification/regression tasks across the 7 v1 datasets
    assert len(specs) == 21
    assert {s.dataset for s in specs} == set(RELBENCH_V1_DATASETS)
    # no link-prediction leaked in
    assert all(s.task_type != TaskType.LINK_PREDICTION for s in specs)
