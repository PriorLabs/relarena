"""RelGNN baselines with full-data and early-stopping refit policies."""

from relarena.models.relgnn.model import (
    RELGNN_SPACE,
    RelGNNEarlyStopModel,
    RelGNNModel,
)

__all__ = ["RELGNN_SPACE", "RelGNNEarlyStopModel", "RelGNNModel"]
