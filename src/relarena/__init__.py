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

from relarena.cache import CacheConfig, CacheMiss, cache_key, cached_artifact
from relarena.checksums import (
    database_checksum,
    split_checksums,
    table_checksum,
)
from relarena.dataset import InnerSplit, OuterSplit, RelBenchDatasetTask, Split
from relarena.identity import RunIdentity
from relarena.model import RelArenaModel
from relarena.registry import ModelRegistry, register_model, registry
from relarena.results import TrialResult, summary_to_dataframe
from relarena.runner import run_experiment
from relarena.tasks import RELBENCH_V1_DATASETS, TaskSpec, list_entity_tasks
from relarena.tuner import tune

__all__ = [
    "RELBENCH_V1_DATASETS",
    "CacheConfig",
    "CacheMiss",
    "RelArenaModel",
    "RunIdentity",
    "ModelRegistry",
    "register_model",
    "registry",
    "RelBenchDatasetTask",
    "Split",
    "InnerSplit",
    "OuterSplit",
    "TaskSpec",
    "TrialResult",
    "cache_key",
    "cached_artifact",
    "list_entity_tasks",
    "run_experiment",
    "summary_to_dataframe",
    "tune",
    "table_checksum",
    "database_checksum",
    "split_checksums",
]
__version__ = "0.0.2"
