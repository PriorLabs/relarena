"""Train-only target rules: lattice, log scale, and zero floor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.nori_rel import targets as targets_module
from relarena.models.nori_rel.targets import TargetRules

ALL_ON = {"lattice": True, "skew_transform": True, "nonnegative": True}


@pytest.mark.parametrize(
    ("values", "task_type", "expected"),
    [
        ([3.0, 1.0, 2.0], TaskType.REGRESSION, (1.0, 2.0, 3.0)),
        ([1.0, 2.5, 3.0], TaskType.REGRESSION, None),
        ([1.0] * 3, TaskType.REGRESSION, None),
        (list(range(51)), TaskType.REGRESSION, None),
        ([0.0, 1.0], TaskType.BINARY_CLASSIFICATION, None),
    ],
)
def test_lattice_is_train_only_and_bounded(
    values: list[float], task_type: TaskType, expected: tuple[float, ...] | None
) -> None:
    target = pd.Series(values)

    assert TargetRules.fit(target, task_type, **ALL_ON).lattice == expected
    off = TargetRules.fit(
        target, task_type, lattice=False, skew_transform=False, nonnegative=False
    )
    assert off == TargetRules()


@pytest.mark.parametrize(
    ("values", "task_type", "expected"),
    [
        ([0.0] * 99 + [1000.0], TaskType.REGRESSION, "log1p"),
        ([1.0, 2.0, 3.0], TaskType.REGRESSION, None),
        ([-1.0, 0.0, 1000.0], TaskType.REGRESSION, None),
        ([0.0] * 99 + [1000.0], TaskType.BINARY_CLASSIFICATION, None),
    ],
)
def test_skew_transform_is_train_only_and_fails_closed(
    values: list[float], task_type: TaskType, expected: str | None
) -> None:
    rules = TargetRules.fit(
        pd.Series(values),
        task_type,
        lattice=False,
        skew_transform=True,
        nonnegative=False,
    )

    assert rules.transform == expected


def test_lattice_takes_precedence_over_the_skew_transform() -> None:
    rules = TargetRules.fit(
        pd.Series([0.0] * 99 + [1000.0]),
        TaskType.REGRESSION,
        lattice=True,
        skew_transform=True,
        nonnegative=False,
    )

    assert rules.lattice == (0.0, 1000.0)
    assert rules.discretization == "snap-median"
    assert rules.transform is None


def test_large_log_target_requires_heavy_zero_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(targets_module, "LOG_TARGET_MAX_ROWS", 50)
    skewed = {"lattice": False, "skew_transform": True, "nonnegative": False}

    zero_heavy = TargetRules.fit(
        pd.Series([0.0] * 99 + [1000.0]), TaskType.REGRESSION, **skewed
    )
    positive = TargetRules.fit(
        pd.Series([1.0] * 99 + [1000.0]), TaskType.REGRESSION, **skewed
    )

    assert zero_heavy.transform == "log1p"
    assert positive.transform is None


@pytest.mark.parametrize(
    ("values", "task_type", "expected"),
    [
        ([0.0, 1.0, 3.0], TaskType.REGRESSION, 0.0),
        ([1.0, 2.0, 3.0], TaskType.REGRESSION, None),
        ([-1.0, 0.0, 3.0], TaskType.REGRESSION, None),
        ([0.0, 1.0], TaskType.BINARY_CLASSIFICATION, None),
    ],
)
def test_zero_floor_is_train_only_and_fails_closed(
    values: list[float], task_type: TaskType, expected: float | None
) -> None:
    rules = TargetRules.fit(
        pd.Series(values),
        task_type,
        lattice=False,
        skew_transform=False,
        nonnegative=True,
    )

    assert rules.lower_bound == expected


def test_log_transform_round_trips_and_applies_the_floor() -> None:
    rules = TargetRules(transform="log1p", lower_bound=0.0)
    target = pd.Series([0.0, 1.0, 9.0])

    np.testing.assert_allclose(rules.inverse(rules.forward(target).to_numpy()), target)
    np.testing.assert_array_equal(rules.inverse(np.array([-1.0])), [0.0])
    np.testing.assert_array_equal(
        TargetRules(lower_bound=0.0).inverse(np.array([-0.5, 0.5])), [0.0, 0.5]
    )
