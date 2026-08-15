"""`relgnn` — composite message passing over atomic routes.

Importing this package registers `relgnn` and `relgnn-es` (scored at its best-val
checkpoint instead of a train+val refit). See `model`; `_vendor` holds the
upstream RelGNN stack.
"""

from relarena.models.relgnn.model import (
    RELGNN_SPACE,
    RelGNNEarlyStopModel,
    RelGNNModel,
)

__all__ = [
    "RELGNN_SPACE",
    "RelGNNEarlyStopModel",
    "RelGNNModel",
]
