"""RelArena: open, reproducible benchmarking for relational learning.

This alpha release is a living benchmark focused on RelBench v1 entity-level
forecasting tasks.

Design principles (adapted from TabArena, https://tabarena.ai):
  * One model contract every method implements (`RelArenaModel`).
  * The harness owns the tuning and evaluation procedure; callers supply the
    method-specific budget, and models supply training code plus a search space.
  * Every trial records configurations, metrics, phase timings, and optional
    predictions as useful metadata for later analysis.
"""

from relarena.checksums import (
    database_checksum,
    split_checksums,
    table_checksum,
)
from relarena.core.cache import CacheConfig, CacheMiss, cache_key, cached_artifact
from relarena.core.dataset import InnerSplit, OuterSplit, Split
from relarena.core.identity import RunIdentity
from relarena.core.model import RelArenaModel
from relarena.core.registry import (
    MethodRegistry,
    ModelRegistry,
    register_model,
    register_system,
    registry,
)
from relarena.core.results import SystemResult, TrialResult
from relarena.core.system import RelArenaSystem
from relarena.core.tuner import tune
from relarena.dataset import RelBenchDatasetTask
from relarena.results import summary_to_dataframe
from relarena.runner import (
    run_experiment,
    run_model_experiment,
    run_system_experiment,
)
from relarena.tasks import RELBENCH_V1_DATASETS, TaskSpec, list_entity_tasks

__all__ = [
    "RELBENCH_V1_DATASETS",
    "CacheConfig",
    "CacheMiss",
    "RelArenaModel",
    "RelArenaSystem",
    "RunIdentity",
    "ModelRegistry",
    "MethodRegistry",
    "register_model",
    "register_system",
    "registry",
    "RelBenchDatasetTask",
    "Split",
    "InnerSplit",
    "OuterSplit",
    "TaskSpec",
    "TrialResult",
    "SystemResult",
    "cache_key",
    "cached_artifact",
    "list_entity_tasks",
    "run_experiment",
    "run_model_experiment",
    "run_system_experiment",
    "summary_to_dataframe",
    "tune",
    "table_checksum",
    "database_checksum",
    "split_checksums",
]
__version__ = "0.0.1"
