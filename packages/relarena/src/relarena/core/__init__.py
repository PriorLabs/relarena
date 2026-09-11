"""Shared contracts and runtime for relational prediction."""

from relarena.core.cache import CacheConfig, CacheMiss, cache_key, cached_artifact
from relarena.core.dataset import InnerSplit, OuterSplit, Split, TaskSource
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
