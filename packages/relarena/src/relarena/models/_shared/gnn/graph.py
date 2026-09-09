"""Materializing a RelBench database into a PyG hetero graph.

Shared by every GNN baseline: `graphsage` and `relgt` memoize `build_graph` through
`GRAPH_CACHE`, while `relgnn` keeps its own memo over a disk-backed loader — the
payloads differ, so the caches are deliberately separate instances.

Heavy imports (PyG, PyTorch Frame) stay inside the function so importing a GNN
model to register it does not need the `rdl` extra.
"""

from __future__ import annotations

from typing import Any

from relbench.base import Database

from relarena.models._shared.gnn.graph_cache import DBGraphCache

#: Minibatch size for the GloVe text embedder while materializing the graph.
TEXT_EMBED_BATCH_SIZE = 256

GRAPH_CACHE = DBGraphCache()


def build_graph(db: Database, device: str) -> tuple[Any, dict[str, Any]]:
    """Materialize `db` into a PyG hetero graph (text columns embedded with GloVe)."""
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from relarena.models._shared.gnn._vendor.gnn import GloveTextEmbedding

    return make_pkey_fkey_graph(
        db,
        col_to_stype_dict=get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=TEXT_EMBED_BATCH_SIZE,
        ),
    )
