# SPDX-License-Identifier: MIT
# Copyright (c) 2023 RelBench Team
# Vendored from the relbench 2.1.2 release on PyPI; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelBench stype proposal and pkey-fkey graph construction.

STATUS: vendored from upstream with two semantic changes.
  * Source: relbench 2.1.2 (https://pypi.org/project/relbench/2.1.2/)
    * ``propose_stypes``       <- ``relbench/modeling/utils.py::get_stype_proposal``
    * ``make_pkey_fkey_graph`` <- ``relbench/modeling/graph.py::make_pkey_fkey_graph``
  * Modified:
    * ``propose_stypes`` samples the rows with a fixed seed (``STYPE_SAMPLE_SEED``),
      as in snap-stanford/relbench#431, which is not in 2.1.2.
    * ``make_pkey_fkey_graph`` takes an optional ``col_stats_dict`` and materializes
      each table with a copy of it instead of computing the stats from the table.
    * Heavy imports (PyTorch Frame, PyG) moved inside the functions, so importing a
      GNN model to register it does not need the ``rdl`` extra. Type annotations use
      builtin generics.

To re-sync: re-copy from the pinned files, redo the changes above, and bump the
pinned version above.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any

from relbench.base import Database

if TYPE_CHECKING:
    from torch_frame.config.text_embedder import TextEmbedderConfig

#: Rows sampled per table for stype inference, and the sample's fixed seed.
STYPE_SAMPLE_SIZE = 1_000
STYPE_SAMPLE_SEED = 0


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


def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: dict[str, dict[str, Any]],
    text_embedder_cfg: TextEmbedderConfig | None = None,
    cache_dir: str | None = None,
    col_stats_dict: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a hetero graph with primary-foreign key edges, plus each table's stats.

    Adapted from `relbench.modeling.graph.make_pkey_fkey_graph` (RelBench 2.1.2, MIT),
    which always computes the column stats from the tables it is given. Pass
    `col_stats_dict` to materialize with existing stats instead: torch_frame indexes
    categories by frequency, so a later snapshot's own stats renumber them, and the
    encoders trained on the fit graph would read the wrong category behind an index.
    Categories unseen at fit get torch_frame's missing value (-1). Like upstream, this
    pops the key columns from `col_to_stype_dict`.
    """
    import numpy as np
    import pandas as pd
    import torch
    from relbench.modeling.utils import remove_pkey_fkey, to_unix_time
    from torch_frame import stype
    from torch_frame.data import Dataset
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import sort_edge_index

    data = HeteroData()
    graph_col_stats = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        # `materialize` keeps the stats dict it is given and writes `EMB_DIM` into it
        # for text columns, so it gets a copy.
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(
            path=path,
            col_stats=(
                None
                if col_stats_dict is None
                else copy.deepcopy(col_stats_dict[table_name])
            ),
        )

        data[table_name].tf = dataset.tensor_frame
        graph_col_stats[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data, graph_col_stats
