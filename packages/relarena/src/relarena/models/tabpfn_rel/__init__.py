"""`tabpfn-rel` — DFS + a TabPFN foundation model with opt-in feature/context extras.

A config-driven extension of the `rdblearn` recipe (DFS features -> a TFM) adding
calendar / history-lag / text features and recency-aware in-context sampling.
Importing this module registers the `tabpfn-rel-local` and `tabpfn-rel-client`
models. See `relarena.models.tabpfn_rel.model`.
"""

from relarena.models.tabpfn_rel.model import (
    TabPFNRelClientModel,
    TabPFNRelLocalModel,
)

__all__ = ["TabPFNRelClientModel", "TabPFNRelLocalModel"]
