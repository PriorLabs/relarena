"""Vendored RelGNN model code (snap-stanford/RelGNN), kept verbatim.

The relarena adapter (fit/predict, disk graph cache, search space) lives in
`relarena.models.relgnn.model`. Only the dependency-free `get_atomic_routes` is
re-exported here; `RelGNN_Model` in `model.py` pulls the heavy GNN stack (PyG /
PyTorch Frame), so the adapter imports it lazily inside `fit` — mirroring how
`graphsage` imports the vendored `Model` from `_shared/gnn/_vendor/gnn.py`.
"""

from relarena.models.relgnn._vendor.atomic_routes import get_atomic_routes

__all__ = ["get_atomic_routes"]
