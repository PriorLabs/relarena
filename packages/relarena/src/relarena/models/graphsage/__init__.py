"""`graphsage` — RelBench`s heterogeneous, temporal GraphSAGE GNN.

Importing this package registers the `graphsage` model. See `model`; the vendored
GNN building blocks it drives live in `models/_shared/gnn/_vendor`.
"""

from relarena.models.graphsage.model import GRAPHSAGE_SPACE, GraphSAGEModel

__all__ = ["GRAPHSAGE_SPACE", "GraphSAGEModel"]
