"""Shared contracts and runtime for relational prediction."""

from relarena_core.cache import CacheConfig, CacheMiss, cache_key, cached_artifact
from relarena_core.dataset import InnerSplit, OuterSplit, Split, TaskSource
from relarena_core.discovery import discover_models
from relarena_core.identity import RunIdentity
from relarena_core.model import RelArenaModel
from relarena_core.registry import (
    MethodRegistry,
    ModelRegistry,
    register_model,
    register_system,
    registry,
)
from relarena_core.results import SystemResult, TrialResult
from relarena_core.system import RelArenaSystem
from relarena_core.tuner import tune

__version__ = "0.0.1"
__all__ = [
    "CacheConfig",
    "CacheMiss",
    "cache_key",
    "cached_artifact",
    "TaskSource",
    "Split",
    "InnerSplit",
    "OuterSplit",
    "discover_models",
    "RunIdentity",
    "RelArenaModel",
    "RelArenaSystem",
    "MethodRegistry",
    "ModelRegistry",
    "register_model",
    "register_system",
    "registry",
    "TrialResult",
    "SystemResult",
    "tune",
]
