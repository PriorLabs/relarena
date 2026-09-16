# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2023 RelBench Team
# Vendored from snap-stanford/RelGNN @ cffdb8b; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGNN per-route conv (``RelGNNConv``).

STATUS: vendored from upstream; kept verbatim (no semantic change).
  * Source: snap-stanford/RelGNN @ commit cffdb8b54627e92c7dd112c1243dde739c90d35b
    https://github.com/snap-stanford/RelGNN/blob/cffdb8b54627e92c7dd112c1243dde739c90d35b/examples/relgnn_conv.py

A ``TransformerConv`` subclass that runs attention along an atomic route: for a
``dim-dim`` route it attends source->destination directly; for a ``dim-fact-dim`` route
it first SAGE-aggregates the fact table into the attention source, then attends.

Ruff-exempt (``# ruff: noqa``) so it stays a clean diff against upstream. To re-sync:
re-copy from the pinned file and bump the commit above.
"""

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import TransformerConv, SAGEConv


class RelGNNConv(TransformerConv):
    def __init__(
        self,
        attn_type,
        in_channels,
        out_channels,
        heads,
        aggr,
        simplified_MP=False,
        bias=True,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, heads, bias=bias, **kwargs)
        self.attn_type = attn_type
        if attn_type == "dim-fact-dim":
            self.aggr_conv = SAGEConv(in_channels, out_channels, aggr=aggr)
        self.simplified_MP = simplified_MP
        self.final_proj = Linear(heads * out_channels, out_channels, bias=bias)
        self.final_proj.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        return_attention_weights=None,
    ):
        # dim-dim
        if self.attn_type == "dim-dim":
            if self.simplified_MP and edge_index.shape[1] == 0:
                return None
            out = super().forward(x, edge_index, edge_attr, return_attention_weights)
            return self.final_proj(out)

        # dim-fact-dim
        edge_attn, edge_aggr = edge_index

        src_aggr, dst_aggr, dst_attn = x

        if self.simplified_MP:
            if edge_attn.shape[1] == 0:
                return None

            if edge_aggr.shape[1] == 0:
                src_attn = dst_aggr
            else:
                src_attn = self.aggr_conv((src_aggr, dst_aggr), edge_aggr)
        else:
            src_attn = self.aggr_conv((src_aggr, dst_aggr), edge_aggr)

        out = super().forward(
            (src_attn, dst_attn), edge_attn, edge_attr, return_attention_weights
        )

        return self.final_proj(out), src_attn
