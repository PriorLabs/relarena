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
fit graph's stypes (`build_predict_graph`), so its column encoders see the layout they
were trained on.
"""

from __future__ import annotations

import copy
from typing import Any

from relbench.base import Database

from relarena.models._shared.gnn.graph_cache import DBGraphCache

#: Minibatch size for the GloVe text embedder while materializing the graph.
TEXT_EMBED_BATCH_SIZE = 256

#: Rows sampled per table for stype inference, and the sample's fixed seed.
STYPE_SAMPLE_SIZE = 1_000
STYPE_SAMPLE_SEED = 0

#: Instance attribute on a built graph holding the stypes it was materialized with.
_STYPES_ATTR = "_relarena_col_to_stype"

GRAPH_CACHE = DBGraphCache()


def propose_stypes(db: Database) -> dict[str, dict[str, Any]]:
    """Infer each table's column stypes from a seeded row sample.

    Adapted from `relbench.modeling.utils.get_stype_proposal` (RelBench 2.1.2, MIT),
    which samples the rows with an unseeded `df.sample`: the same database can then
    get a different proposal on every call (e.g. a low-cardinality integer column
    flips between categorical and numerical).
    """
    from torch_frame import stype
    from torch_frame.utils import infer_df_stype

    proposal: dict[str, dict[str, Any]] = {}
    for table_name, table in db.table_dict.items():
        df = table.df.sample(
            min(STYPE_SAMPLE_SIZE, len(table.df)), random_state=STYPE_SAMPLE_SEED
        )
        col_to_stype = infer_df_stype(df)
        # As upstream: RelBench's encoders have no embedding stype (rel-amazon).
        for col, col_stype in col_to_stype.items():
            if col_stype == stype.embedding:
                col_to_stype[col] = stype.multicategorical
        proposal[table_name] = col_to_stype
    return proposal


def graph_col_to_stype(data: Any) -> dict[str, dict[str, Any]] | None:
    """Return the stypes `data` was materialized with by `build_graph`, else None."""
    return getattr(data, _STYPES_ATTR, None)


def build_graph(
    db: Database,
    device: str,
    col_to_stype_dict: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Materialize `db` into a PyG hetero graph (text columns embedded with GloVe).

    `col_to_stype_dict` defaults to `propose_stypes(db)`; pass `graph_col_to_stype`
    of the fit graph when materializing a different database for prediction.
    """
    from relbench.modeling.graph import make_pkey_fkey_graph
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from relarena.models._shared.gnn._vendor.gnn import GloveTextEmbedding

    if col_to_stype_dict is None:
        col_to_stype_dict = propose_stypes(db)
    # `make_pkey_fkey_graph` pops key columns from the dict it is given; keep ours.
    stypes = copy.deepcopy(col_to_stype_dict)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=copy.deepcopy(col_to_stype_dict),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=TEXT_EMBED_BATCH_SIZE,
        ),
    )
    # A plain instance attribute, not a PyG store entry: loaders and `.to()` skip it.
    object.__setattr__(data, _STYPES_ATTR, stypes)
    return data, col_stats_dict


def build_predict_graph(
    db: Database, device: str, fit_col_to_stype: dict[str, dict[str, Any]] | None
) -> tuple[Any, dict[str, Any]]:
    """Materialize a predict-time `db` with the fit graph's stypes, when known.

    `fit_col_to_stype` is `graph_col_to_stype` of the graph the model was fit on;
    None (a graph not built by `build_graph`) falls back to a fresh proposal.
    """
    if fit_col_to_stype is None:
        return build_graph(db, device)
    return build_graph(db, device, col_to_stype_dict=fit_col_to_stype)
