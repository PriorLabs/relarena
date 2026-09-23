"""Training-label profile for generic regression routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from relbench.base import TaskType

ENDPOINT_FRACTION_THRESHOLD: Final = 0.85
LOW_SKEW_THRESHOLD: Final = 1.0
MIN_LEVELS_EXCLUSIVE: Final = 50
BOUNDARY_ATOL: Final = 1e-10


@dataclass(frozen=True)
class TargetProfile:
    """Aggregate target facts learned without retaining training labels."""

    endpoint_rich_bounded: bool = False
    low_skew_many_level: bool = False
    endpoint_fraction: float = 0.0
    distinct_levels: int = 0
    skew: float = float("nan")

    @property
    def use_depth_three(self) -> bool:
        """Whether this training target activates the deeper relational search."""
        return self.endpoint_rich_bounded

    @classmethod
    def fit(cls, target: pd.Series, task_type: TaskType) -> TargetProfile:
        """Compute fail-closed regression gates from the training target."""
        if task_type != TaskType.REGRESSION:
            return cls()

        values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            return cls()

        endpoints = np.isclose(values, 0.0, rtol=0.0, atol=BOUNDARY_ATOL) | np.isclose(
            values,
            1.0,
            rtol=0.0,
            atol=BOUNDARY_ATOL,
        )
        endpoint_fraction = float(endpoints.mean())
        bounded = bool(
            values.min() >= -BOUNDARY_ATOL and values.max() <= 1.0 + BOUNDARY_ATOL
        )
        distinct_levels = int(pd.unique(values).size)
        skew = float(pd.Series(values).skew())
        low_skew_many_level = bool(
            distinct_levels > MIN_LEVELS_EXCLUSIVE
            and np.isfinite(skew)
            and abs(skew) <= LOW_SKEW_THRESHOLD
        )
        return cls(
            endpoint_rich_bounded=(
                bounded and endpoint_fraction >= ENDPOINT_FRACTION_THRESHOLD
            ),
            low_skew_many_level=low_skew_many_level,
            endpoint_fraction=endpoint_fraction,
            distinct_levels=distinct_levels,
            skew=skew,
        )
