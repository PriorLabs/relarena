"""Native, backend-neutral relational feature generation."""

from .backend import NoriDFS
from .contracts import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    FeatureSpec,
    ForeignKey,
    NativeFeatureConfig,
    NativeFeatureMatrix,
    PathDirection,
    RelationPath,
)
from .engine import NativeRelationalFeaturizer

__all__ = [
    "AnchorDefinition",
    "DatabaseView",
    "FeaturePlan",
    "FeatureSpec",
    "ForeignKey",
    "NativeFeatureConfig",
    "NativeFeatureMatrix",
    "NativeRelationalFeaturizer",
    "NoriDFS",
    "PathDirection",
    "RelationPath",
]
