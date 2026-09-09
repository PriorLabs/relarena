# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2023 RelBench Team
# Vendored from snap-stanford/RelGNN @ cffdb8b; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGNN message-passing stack (``RelGNN``).

STATUS: vendored from upstream; semantic changes limited to imports + a drop.
  * Source: snap-stanford/RelGNN @ commit cffdb8b54627e92c7dd112c1243dde739c90d35b
    https://github.com/snap-stanford/RelGNN/blob/cffdb8b54627e92c7dd112c1243dde739c90d35b/examples/relgnn_nn.py
  * Modified: the ``RelGNNConv`` / ``RelGNN_HeteroConv`` imports are rewritten to the
    ``relarena.models.relgnn._vendor`` package paths. Upstream's ``HeteroEncoder`` /
    ``HeteroTemporalEncoder`` are NOT copied here: ``RelGNN_Model`` (model.py) imports
    those from the installed ``relbench.modeling.nn`` (they duplicate it), so only the
    ``RelGNN`` class is taken. Everything else is upstream (ruff formatting only).

Ruff-exempt (``# ruff: noqa``) so it stays a clean diff against upstream. To re-sync:
re-copy the ``RelGNN`` class from the pinned file, redo the import rewrites, bump the
commit above.
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.nn import LayerNorm
from torch_geometric.typing import EdgeType, NodeType

from relarena.models.relgnn._vendor.conv import RelGNNConv
from relarena.models.relgnn._vendor.hetero_conv import RelGNN_HeteroConv


class RelGNN(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "sum",
        num_model_layers: int = 2,
        num_heads: int = 1,
        simplified_MP=False,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_model_layers):
            conv = RelGNN_HeteroConv(
                {
                    edge_type: RelGNNConv(
                        edge_type[0],
                        (channels, channels),
                        channels,
                        num_heads,
                        aggr=aggr,
                        simplified_MP=simplified_MP,
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
                simplified_MP=simplified_MP,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_model_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
