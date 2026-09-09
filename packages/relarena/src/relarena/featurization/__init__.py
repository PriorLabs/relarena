"""Featurization: relational database -> a flat feature table (for tabular models).

Sub-modules implement different recipes:
  * `entity` — entity-only (the RelBench LightGBM recipe);
  * `dfs`    — multi-hop Deep Feature Synthesis (the RDBLearn recipe), with a
    depth cache.

Both expose a `build_*_features(...) -> (features_df, categorical_columns)`
function; shared column typing lives in `_columns`.
"""

from relarena.featurization.dfs import DFS_MAX_DEPTH, build_dfs_features
from relarena.featurization.entity import build_entity_features

__all__ = ["DFS_MAX_DEPTH", "build_entity_features", "build_dfs_features"]
