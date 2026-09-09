"""Relational predictive queries over user-supplied databases."""

from relarena.core.userdb.ingest import DatabaseSpec
from relarena.core.userdb.query import PredictiveQuery, PredictiveQuerySpec
from relarena.core.userdb.spec import PredictiveTaskSpec

__all__ = [
    "DatabaseSpec",
    "PredictiveQuery",
    "PredictiveQuerySpec",
    "PredictiveTaskSpec",
]
