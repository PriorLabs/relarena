"""Train-only target rules for NoriDFS regression: lattice, log scale, floor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from relbench.base import TaskType

TARGET_LATTICE_MAX_LEVELS: Final = 50
EXTREME_TARGET_SKEW: Final = 8.0
LOG_TARGET_ZERO_FRACTION: Final = 0.75
LOG_TARGET_MAX_ROWS: Final = 60_000
LOG1P: Final = "log1p"
SNAP_MEDIAN: Final = "snap-median"


@dataclass(frozen=True)
class TargetRules:
    """Fit-frozen decisions about how Nori sees the target and returns it."""

    lattice: tuple[float, ...] | None = None
    transform: str | None = None
    lower_bound: float | None = None

    @property
    def discretization(self) -> str | None:
        """Return Nori's decoder name when the target is a small integer lattice."""
        return SNAP_MEDIAN if self.lattice is not None else None

    @classmethod
    def fit(
        cls,
        target: pd.Series,
        task_type: TaskType,
        *,
        lattice: bool,
        skew_transform: bool,
        nonnegative: bool,
    ) -> TargetRules:
        """Derive every rule from the fitted labels alone; each gate fails closed."""
        if task_type != TaskType.REGRESSION:
            return cls()
        values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        levels = _lattice(values) if lattice else None
        transform = (
            _skew_transform(values) if skew_transform and levels is None else None
        )
        lower_bound = _lower_bound(values) if nonnegative else None
        return cls(lattice=levels, transform=transform, lower_bound=lower_bound)

    def forward(self, target: pd.Series) -> pd.Series:
        """Move fitted labels onto the scale Nori is trained on."""
        if self.transform is None:
            return target
        if self.transform == LOG1P:
            return np.log1p(target.astype(float))
        raise ValueError(f"unknown target transform {self.transform!r}")

    def inverse(self, prediction: np.ndarray) -> np.ndarray:
        """Return predictions to the label scale and apply the fitted floor."""
        if self.transform == LOG1P:
            prediction = np.maximum(0.0, np.expm1(prediction))
        elif self.transform is not None:
            raise ValueError(f"unknown target transform {self.transform!r}")
        if self.lower_bound is not None:
            prediction = np.maximum(self.lower_bound, prediction)
        return prediction


def _lattice(values: np.ndarray) -> tuple[float, ...] | None:
    """Freeze a small integer lattice when every fitted label sits on one."""
    if not np.isfinite(values).all() or not np.allclose(
        values, np.rint(values), rtol=0.0, atol=1e-10
    ):
        return None
    levels = np.unique(values)
    if not 2 <= len(levels) <= TARGET_LATTICE_MAX_LEVELS:
        return None
    return tuple(float(level) for level in levels)


def _skew_transform(values: np.ndarray) -> str | None:
    """Use log1p for extreme skew on small or heavily zero-inflated targets."""
    if len(values) < 3 or not np.isfinite(values).all() or values.min() < 0:
        return None
    skew = float(pd.Series(values).skew())
    zero_fraction = float(np.mean(values == 0.0))
    supported = len(values) <= LOG_TARGET_MAX_ROWS or (
        zero_fraction >= LOG_TARGET_ZERO_FRACTION
    )
    if supported and np.isfinite(skew) and skew >= EXTREME_TARGET_SKEW:
        return LOG1P
    return None


def _lower_bound(values: np.ndarray) -> float | None:
    """Infer a zero floor only when the fitted labels demonstrate it."""
    if (
        len(values) < 2
        or not np.isfinite(values).all()
        or values.min() != 0.0
        or values.max() <= 0.0
    ):
        return None
    return 0.0


__all__ = ["TargetRules"]
