"""Typed view of the NoriDFS configuration keys accepted by the adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .context import RESERVOIR_RANDOM, RESERVOIRS

TRAIN_PROFILE_NONE: Final = "none"
TRAIN_PROFILE_V1: Final = "v1"
TRAIN_PROFILES: Final = frozenset({TRAIN_PROFILE_NONE, TRAIN_PROFILE_V1})
CONTEXT_PROFILE_NONE: Final = "none"
CONTEXT_PROFILE_V1: Final = "v1"
CONTEXT_PROFILES: Final = frozenset({CONTEXT_PROFILE_NONE, CONTEXT_PROFILE_V1})
#: Entities with at least this many training rows carry one lag, not five.
DENSE_HISTORY_ROWS_PER_ENTITY: Final = 32
HISTORY_LAGS: Final = 5


@dataclass(frozen=True)
class NoriDFSRecipe:
    """Every NoriDFS knob, read once from the RelArena config and validated."""

    max_paths: int = 24
    max_features: int = 48
    max_direct_features: int = 1
    max_categorical_cardinality: int = 32
    max_direct_categorical_cardinality: int | None = None
    windows: tuple[int, ...] = (1, 4, 16)
    target_history: bool = True
    train_profile: str = TRAIN_PROFILE_NONE
    dense_history: bool = False
    target_lattice: bool = False
    skew_target_transform: bool = False
    nonnegative_support: bool = False
    reservoir: str = RESERVOIR_RANDOM
    context_profile: str = CONTEXT_PROFILE_NONE
    missing_category_modes: bool = False
    missing_fractions: bool = False
    memory_policy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject values the planner or the fit path cannot honour."""
        if self.max_paths < 1 or self.max_features < 1 or self.max_direct_features < 0:
            raise ValueError("noridfs budgets must be positive")
        if not self.windows or any(window < 1 for window in self.windows):
            raise ValueError("noridfs_windows must contain positive integers")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("noridfs_windows must not contain duplicates")
        if self.train_profile not in TRAIN_PROFILES:
            raise ValueError(f"unknown train_profile {self.train_profile!r}")
        if self.reservoir not in RESERVOIRS:
            raise ValueError(f"unknown noridfs_train_reservoir {self.reservoir!r}")
        if self.context_profile not in CONTEXT_PROFILES:
            raise ValueError(
                f"unknown noridfs_context_profile {self.context_profile!r}"
            )

    @property
    def uses_history_lags(self) -> bool:
        """Whether the train profile appends strict-past target lags."""
        return self.train_profile == TRAIN_PROFILE_V1

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NoriDFSRecipe:
        """Read the ``noridfs_*`` keys, keeping the fixed-configuration defaults."""
        cardinality = config.get("noridfs_max_direct_categorical_cardinality")
        return cls(
            max_paths=int(config.get("noridfs_max_paths", 24)),
            max_features=int(config.get("noridfs_max_features", 48)),
            max_direct_features=int(config.get("noridfs_max_direct_features", 1)),
            max_categorical_cardinality=int(
                config.get("noridfs_max_categorical_cardinality", 32)
            ),
            max_direct_categorical_cardinality=(
                None if cardinality is None else int(cardinality)
            ),
            windows=_windows(config.get("noridfs_windows", "1,4,16")),
            target_history=bool(config.get("noridfs_target_history", True)),
            train_profile=str(config.get("train_profile", TRAIN_PROFILE_NONE)),
            dense_history=bool(config.get("noridfs_dense_history", False)),
            target_lattice=bool(config.get("noridfs_target_lattice", False)),
            skew_target_transform=bool(
                config.get("noridfs_skew_target_transform", False)
            ),
            nonnegative_support=bool(config.get("noridfs_nonnegative_support", False)),
            reservoir=str(config.get("noridfs_train_reservoir", RESERVOIR_RANDOM)),
            context_profile=str(
                config.get("noridfs_context_profile", CONTEXT_PROFILE_NONE)
            ),
            missing_category_modes=bool(
                config.get("noridfs_missing_category_modes", False)
            ),
            missing_fractions=bool(config.get("noridfs_missing_fractions", False)),
            memory_policy=config.get("nori_memory_policy"),
        )

    def history_lags(self, median_rows_per_entity: float | None) -> int:
        """Return the lag count, compressing five to one for dense histories."""
        if not self.uses_history_lags:
            return 0
        if (
            self.dense_history
            and median_rows_per_entity is not None
            and median_rows_per_entity >= DENSE_HISTORY_ROWS_PER_ENTITY
        ):
            return 1
        return HISTORY_LAGS


def _windows(value: object) -> tuple[int, ...]:
    """Parse the comma-separated window multipliers used in config identities."""
    try:
        return tuple(int(part) for part in str(value).split(","))
    except ValueError as exc:
        raise ValueError("noridfs_windows must be comma-separated integers") from exc


__all__ = [
    "CONTEXT_PROFILE_V1",
    "DENSE_HISTORY_ROWS_PER_ENTITY",
    "TRAIN_PROFILE_V1",
    "NoriDFSRecipe",
]
