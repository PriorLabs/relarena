import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.nori_rel.profile import TargetProfile


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0] * 85 + [0.5] * 15, True),
        ([1.0] * 84 + [0.5] * 16, False),
        ([0.0] * 85 + [1.1] * 15, False),
    ],
)
def test_endpoint_route_is_bounded_and_meets_the_training_fraction(
    values: list[float],
    expected: bool,
) -> None:
    profile = TargetProfile.fit(pd.Series(values), TaskType.REGRESSION)

    assert profile.endpoint_rich_bounded is expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (list(range(60)), True),
        (list(range(50)), False),
        ([float(value) + 0.25 for value in range(60)], True),
        (list(range(60)) + [10_000] * 500, False),
    ],
)
def test_mean_route_requires_more_than_fifty_low_skew_levels(
    values: list[float],
    expected: bool,
) -> None:
    profile = TargetProfile.fit(pd.Series(values), TaskType.REGRESSION)

    assert profile.low_skew_many_level is expected


def test_profile_fails_closed_for_nonregression_and_nonfinite_targets() -> None:
    binary = TargetProfile.fit(pd.Series([0.0, 1.0]), TaskType.BINARY_CLASSIFICATION)
    nonfinite = TargetProfile.fit(
        pd.Series([0.0, np.nan, 1.0]),
        TaskType.REGRESSION,
    )

    assert binary == TargetProfile()
    assert nonfinite == TargetProfile()
