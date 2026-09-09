"""TabPFN models and a predictive interface for relational tables."""

from relarena_core.userdb import PredictiveQuery, PredictiveQuerySpec

from tabpfn_rel.model import (
    TABPFN_REL_CLIENT_SPACE,
    TABPFN_REL_LOCAL_SPACE,
    TabPFNRelClientModel,
    TabPFNRelLocalModel,
    TabPFNRelModel,
)

__all__ = [
    "TABPFN_REL_CLIENT_SPACE",
    "TABPFN_REL_LOCAL_SPACE",
    "PredictiveQuery",
    "PredictiveQuerySpec",
    "TabPFNRelClientModel",
    "TabPFNRelLocalModel",
    "TabPFNRelModel",
]
__version__ = "0.0.2"
