"""DFS-family view of the shared feature contract.

``BaseDFS`` is the historical name runtime callers import; it is the same
class as ``nori_rel.feature.FeatureBackend``.
"""

from .._contract import CUTOFF_COLUMN, FeatureBackend, FeatureMatrices, RelationalData

BaseDFS = FeatureBackend

__all__ = ["CUTOFF_COLUMN", "BaseDFS", "FeatureMatrices", "RelationalData"]
