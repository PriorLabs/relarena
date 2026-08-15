# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen,
# Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey,
# Jure Leskovec
# Vendored from snap-stanford/relgt @ 19e423ca; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGT model — multi-element tokens + local/global attention (``RelGT``).

STATUS: vendored from upstream; the only semantic change is the imports.
  * Source: snap-stanford/relgt @ commit 19e423ca3e7cac761130aba790857f2dc3a46ef7
    https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/model.py
  * Modified: the three inter-module imports (``codebook``, ``local_module``,
    ``encoders``) are rewritten to ``relarena.models.relgt._vendor.*`` package paths so
    they resolve as a subpackage. Search this file for ``relarena.models.relgt._vendor``
    to see them — those are the only semantic edit.
  * Everything else (every class, method, and default) is semantically upstream;
    ruff normalizes formatting (whitespace/quotes/wrapping), so the rest of the diff
    against upstream is formatting-only.

Ruff-exempt (``# ruff: noqa``) so it stays a clean diff against upstream. relarena
glue (fit/predict, the tokenization adapter, the search space, caches) is NOT here —
it lives in ``relarena.models.relgt`` and ``relarena.models.relgt.tokenize``. To
re-sync: re-copy from the pinned file, redo the three import rewrites, bump the commit.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MLP

from relarena.models.relgt._vendor.codebook import VectorQuantizerEMA
from einops import rearrange
from relarena.models.relgt._vendor.local_module import LocalModule

from torch_frame.data.stats import StatType
from typing import Dict, Any, List

from relarena.models.relgt._vendor.encoders import (
    NeighborNodeTypeEncoder,
    NeighborHopEncoder,
    NeighborTimeEncoder,
    NeighborTfsEncoder,
    GNNPEEncoder,
)


class RelGTLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        local_num_layers,
        global_dim,
        num_nodes,
        heads=1,
        concat=True,
        ff_dropout=0.0,
        attn_dropout=0.0,
        edge_dim=None,
        conv_type="local",
        num_centroids=None,
        sample_node_len=100,
        **kwargs,
    ):
        super(RelGTLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_num_layers = local_num_layers
        self.heads = heads
        self.concat = concat
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None
        self.sample_node_len = sample_node_len

        self.local_module = LocalModule(
            seq_len=self.sample_node_len,
            input_dim=in_channels,
            n_layers=local_num_layers,
            num_heads=heads,
            hidden_dim=out_channels,
            dropout_rate=ff_dropout,
            attention_dropout_rate=attn_dropout,
        )
        self.layer_norm_local = nn.LayerNorm(out_channels)

        if self.conv_type != "local":
            self.vq = VectorQuantizerEMA(num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
            self.register_buffer("c_idx", c)
            self.attn_fn = F.softmax

            attn_channels = out_channels // heads

            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim, heads * attn_channels)
            self.lin_query_g = Linear(global_dim, heads * attn_channels)
            self.lin_value_g = Linear(global_dim, heads * attn_channels)
            self.layer_norm_global = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # Reinitialize global attention layers
        if self.conv_type != "local":
            self.lin_proj_g.reset_parameters()
            self.lin_key_g.reset_parameters()
            self.lin_query_g.reset_parameters()
            self.lin_value_g.reset_parameters()
            if hasattr(self, "vq"):
                self.vq.reset_parameters()

        # Reinitialize LocalModule
        if hasattr(self.local_module, "reset_parameters"):
            self.local_module.reset_parameters()

    def forward(self, x_set, x, node_indices):
        if self.conv_type == "local":
            out = self.local_forward(x_set)
            out = self.layer_norm_local(out)

        elif self.conv_type == "global":
            out = self.global_forward(x, node_indices)
            out = self.layer_norm_global(out)

        elif self.conv_type == "full":
            out_local = self.local_forward(x_set)
            out_global = self.global_forward(x, node_indices)
            out_local = self.layer_norm_local(out_local)
            out_global = self.layer_norm_global(out_global)
            out = torch.cat([out_local, out_global], dim=1)

        else:
            raise NotImplementedError

        return out

    def global_forward(self, x, batch_idx):
        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = self.lin_proj_g(x)

        k_buf = self.vq.get_k()
        k_x = k_buf.detach().clone()
        v_buf = self.vq.get_v()
        v_x = v_buf.detach().clone()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=h), (q, k, v))
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)

        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count

        dots = dots + torch.log(centroid_count.view(1, 1, -1))

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h n d -> n (h d)")

        # Update the centroids
        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)

        return out

    def local_forward(self, x_set, pretrain_token=False):
        return self.local_module(x_set, pretrain_token)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"local_num_layers={self.local_num_layers})"
        )


class RelGT(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        max_neighbor_hop: int,
        node_type_map: Dict[str, int],
        col_names_dict: Dict[str, Dict[str, List[str]]],
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        local_num_layers: int,
        channels: int,
        out_channels: int,
        global_dim: int,
        heads: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        conv_type: str = "full",
        ablate: str = "none",
        gnn_pe_dim: int = 0,
        num_centroids: int = 4096,
        sample_node_len: int = 100,
        args: Any = None,
    ):
        super(RelGT, self).__init__()

        self.max_neighbor_hop = max_neighbor_hop
        self.node_type_map = node_type_map
        num_node_types = len(node_type_map) + 1  # extra element for mask token
        num_hop_types = self.max_neighbor_hop + 1  # extra element for mask token

        self.type_encoder = NeighborNodeTypeEncoder(
            embedding_dim=channels, node_type_map=self.node_type_map
        )
        self.hop_encoder = NeighborHopEncoder(
            embedding_dim=channels, max_neighbor_hop=self.max_neighbor_hop
        )
        self.time_encoder = NeighborTimeEncoder(embedding_dim=channels)
        self.tfs_encoder = NeighborTfsEncoder(
            channels=channels,
            node_type_map=self.node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
        )
        self.pe_encoder = GNNPEEncoder(embedding_dim=channels, pe_dim=gnn_pe_dim)

        self.layer_norm_type = nn.LayerNorm(channels)
        self.layer_norm_hop = nn.LayerNorm(channels)
        self.layer_norm_time = nn.LayerNorm(channels)
        self.layer_norm_tfs = nn.LayerNorm(channels)
        self.layer_norm_pe = nn.LayerNorm(channels)

        hidden_channels = channels

        ablate_key_dict = {"type": 0, "hop": 1, "time": 2, "tfs": 3, "gnn": 4}
        self.ablate_idx = ablate_key_dict.get(ablate, None)
        channel_mult = 5 if self.ablate_idx is None else 4

        self.in_mixture = nn.Sequential(
            nn.Linear(channel_mult * channels, 2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
        )

        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        _overall_num_layers = 1
        for _ in range(_overall_num_layers):
            self.convs.append(
                RelGTLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    local_num_layers=local_num_layers,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                    conv_type=conv_type,
                    num_centroids=num_centroids,
                    sample_node_len=sample_node_len,
                )
            )
            h_times = 2 if conv_type == "full" else 1

            self.ffs.append(
                nn.Sequential(
                    nn.BatchNorm1d(hidden_channels * h_times),  # BN in
                    nn.Linear(h_times * hidden_channels, hidden_channels * 2),
                    nn.GELU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.Dropout(ff_dropout),
                    nn.BatchNorm1d(hidden_channels),  # BN out
                )
            )

        # supervised task head
        self.head = MLP(
            channels,
            hidden_channels=channels,
            out_channels=out_channels,
            num_layers=2,
        )

    def reset_parameters(self):
        self.type_encoder.reset_parameters()
        self.hop_encoder.reset_parameters()
        self.time_encoder.reset_parameters()
        self.tfs_encoder.reset_parameters()
        self.pe_encoder.reset_parameters()

        for layer in self.in_mixture:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            if hasattr(ff, "reset_parameters"):
                ff.reset_parameters()

        self.head.reset_parameters()

    def forward(
        self,
        neighbor_types,
        node_indices,
        neighbor_hops,
        neighbor_times,
        grouped_tf_dict,
        edge_index=None,
        batch=None,
    ):

        neighbor_tfs = self.layer_norm_tfs(
            self.tfs_encoder(grouped_tf_dict, neighbor_types)
        )
        neighbor_types = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
        neighbor_hops = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
        neighbor_times = self.layer_norm_time(self.time_encoder(neighbor_times.float()))
        neighbor_subgraph_pe = self.layer_norm_pe(self.pe_encoder(edge_index, batch))

        cat_list = [
            neighbor_types,
            neighbor_hops,
            neighbor_times,
            neighbor_tfs,
            neighbor_subgraph_pe,
        ]
        if self.ablate_idx is not None:
            cat_list.pop(self.ablate_idx)
        x_set = torch.cat(cat_list, dim=-1)
        x_set = self.in_mixture(x_set)

        x = x_set[:, 0, :]  # select seed token representation
        for i, conv in enumerate(self.convs):
            x_set = conv(x_set, x, node_indices)
            x_set = self.ffs[i](x_set)
        x_set = self.head(x_set)

        return x_set

    def global_forward(self, x, pos_enc, node_indices):
        raise NotImplementedError
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv.global_forward(x, pos_enc, node_indices)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x
