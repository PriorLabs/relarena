"""Relational feature methods, grouped by family.

``dfs`` holds deep feature synthesis backends; ``agentic`` holds agent-written
SQL feature programs. Both implement ``FeatureBackend``. Import a backend from
its own module so optional dependencies stay optional.
"""

from ._contract import CUTOFF_COLUMN, FeatureBackend, FeatureMatrices, RelationalData

__all__ = ["CUTOFF_COLUMN", "FeatureBackend", "FeatureMatrices", "RelationalData"]
