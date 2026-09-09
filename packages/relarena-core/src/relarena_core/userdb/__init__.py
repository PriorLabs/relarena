"""Relational predictive queries over user-supplied databases."""

from relarena_core.userdb.ingest import DatabaseSpec
from relarena_core.userdb.query import PredictiveQuery, PredictiveQuerySpec
from relarena_core.userdb.spec import PredictiveTaskSpec

__all__ = [
    "DatabaseSpec",
    "PredictiveQuery",
    "PredictiveQuerySpec",
    "PredictiveTaskSpec",
]
