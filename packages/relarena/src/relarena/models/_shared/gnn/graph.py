"""Materializing a RelBench database into a PyG hetero graph.

Shared by every GNN baseline: `graphsage` and `relgt` memoize `build_graph` through
`GRAPH_CACHE`, while `relgnn` keeps its own memo over a disk-backed loader — the
payloads differ, so the caches are deliberately separate instances.

Heavy imports (PyG, PyTorch Frame) stay inside the function so importing a GNN
model to register it does not need the `rdl` extra.

Column stypes come from `propose_stypes`, which is deterministic, and every graph
records the stypes it was materialized with (`graph_col_to_stype`). A model that
materializes a second graph at predict time (the RPI path predicts on a database
censored at the query anchor, a different object from the fit database) passes its
fit graph's stypes and column stats to `build_graph`, so its column encoders see the
layout and category indices they were trained on.
"""

from __future__ import annotations

import copy
from typing import Any

from relbench.base import Database

from relarena.models._shared.gnn._vendor.graph import (
    make_pkey_fkey_graph,
    propose_stypes,
)
from relarena.models._shared.gnn.graph_cache import DBGraphCache

#: Minibatch size for the GloVe text embedder while materializing the graph.
TEXT_EMBED_BATCH_SIZE = 256

#: Instance attribute on a built graph holding the stypes it was materialized with.
_STYPES_ATTR = "_relarena_col_to_stype"

GRAPH_CACHE = DBGraphCache()


def graph_col_to_stype(data: Any) -> dict[str, dict[str, Any]] | None:
    """Return the stypes `data` was materialized with by `build_graph`, else None."""
    return getattr(data, _STYPES_ATTR, None)


def build_graph(
    db: Database,
    device: str,
    col_to_stype_dict: dict[str, dict[str, Any]] | None = None,
    col_stats_dict: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Materialize `db` into a PyG hetero graph (text columns embedded with GloVe).

    `col_to_stype_dict` defaults to `propose_stypes(db)`, and the column stats are
    computed from `db` unless `col_stats_dict` is given. When materializing a
    different database for prediction, pass the fit graph's `graph_col_to_stype` and
    column stats.
    """
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from relarena.models._shared.gnn._vendor.gnn import GloveTextEmbedding

    if col_to_stype_dict is None:
        col_to_stype_dict = propose_stypes(db)
    # `make_pkey_fkey_graph` pops key columns from the dict it is given; keep ours.
    stypes = copy.deepcopy(col_to_stype_dict)
    data, graph_col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=copy.deepcopy(col_to_stype_dict),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=TEXT_EMBED_BATCH_SIZE,
        ),
        col_stats_dict=col_stats_dict,
    )
    # A plain instance attribute, not a PyG store entry: loaders and `.to()` skip it.
    object.__setattr__(data, _STYPES_ATTR, stypes)
    return data, graph_col_stats
