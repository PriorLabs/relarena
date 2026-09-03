"""The system contract.

A system owns its complete prediction procedure. Unlike a `RelArenaModel`, it
is not instantiated once per harness-selected configuration: the harness gives
one instance both protocol splits and records the final predictions it returns.
The system may use the inner split for selection in any way it chooses, or not
use it at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from relbench.base import EntityTask, TaskType

from relarena.cache import CacheConfig
from relarena.dataset import InnerSplit, OuterSplit
from relarena.identity import RunIdentity
from relarena.tasks import ENTITY_TASK_TYPES


class RelArenaSystem(ABC):
    """Base contract for an end-to-end relational prediction system."""

    #: Unique, human-readable identifier used in the registry and result tables.
    name: ClassVar[str]

    #: Task types this system can handle.
    supported_task_types: ClassVar[frozenset[TaskType]] = ENTITY_TASK_TYPES

    def __init__(
        self,
        *,
        cache: CacheConfig | None = None,
        run_identity: RunIdentity | None = None,
    ) -> None:
        """Instantiate one system for one complete experiment."""
        self.cache = cache or CacheConfig(directory=None, on_miss="compute")
        self.run_identity = run_identity

    @abstractmethod
    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        """Return predictions aligned with `outer_split.eval_table`.

        The system owns everything needed to produce those predictions,
        including any selection, refitting, ensembling, or pretrained recipe.
        `inner_split` is available for selection but using it is not required.
        Both split objects carry their correctly censored database states, and
        `outer_split` contains no test labels.
        """


__all__ = ["RelArenaSystem"]
