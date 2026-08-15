# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen,
# Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey,
# Jure Leskovec
# Vendored from snap-stanford/relgt @ 19e423ca; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGT model — the five per-token component encoders (type, hop, time, features, PE).

STATUS: vendored 1:1 — NO modifications.
  * Source: snap-stanford/relgt @ commit 19e423ca3e7cac761130aba790857f2dc3a46ef7
    https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/encoders.py
  * Semantically verbatim — no code logic changed. Vs upstream: the leading
    ``# ruff: noqa`` + this docstring are added and ruff normalizes formatting
    (whitespace/quotes/wrapping), so the diff is formatting-only.

Ruff-exempt so it stays a clean diff against upstream. relarena glue (fit/predict, the
tokenization adapter, the search space, caches) is NOT here — it lives in
``relarena.models.relgt`` and ``relarena.models.relgt.tokenize``. To re-sync: re-copy
from the pinned file and bump the commit.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_frame

import torch_geometric.transforms as T
from torch_geometric.data import Data

# ------------------- Encoder classes -------------------- #


class NeighborNodeTypeEncoder(nn.Module):
    """
    Encoder for neighbor types.
    Uses an embedding layer to convert integer type indices into dense vectors.
    """

    def __init__(self, node_type_map, embedding_dim):
        """
        Args:
            node_type_map (dict): A mapping from node type strings to integer indices.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super(NeighborNodeTypeEncoder, self).__init__()
        # Determine the number of unique types from the mapping
        num_types = max(node_type_map.values()) + 1
        self.embedding = nn.Embedding(
            num_embeddings=num_types + 1, embedding_dim=embedding_dim
        )

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, type_indices):
        """
        Args:
            type_indices (Tensor): Tensor of shape (...), containing integer indices for neighbor types.

        Returns:
            Tensor: Embedded representations of shape (..., embedding_dim).
        """
        return self.embedding(type_indices)


class NeighborHopEncoder(nn.Module):
    """
    Encoder for hop distances.
    Uses an embedding layer to convert hop counts into dense vectors.
    """

    def __init__(self, max_neighbor_hop, embedding_dim):
        """
        Args:
            max_neighbor_hop (int): The maximum hop distance in your data.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super(NeighborHopEncoder, self).__init__()
        # +1 because we assume hops start from 0 or 1 and go to max_neighbor_hop inclusive
        self.embedding = nn.Embedding(
            num_embeddings=max_neighbor_hop + 2, embedding_dim=embedding_dim
        )

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, hop_distances):
        """
        Args:
            hop_distances (Tensor): Tensor of shape (...), containing integer hop distances.

        Returns:
            Tensor: Embedded representations of shape (..., embedding_dim).
        """
        shifted = hop_distances + 1
        return self.embedding(shifted)


from torch_geometric.nn import PositionalEncoding


class NeighborTimeEncoder(nn.Module):
    """
    Two-stage time encoder using positional encoding followed by a linear layer.
    """

    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): Dimension of the output embedding.
        """
        super(NeighborTimeEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.mask_vector = nn.Parameter(torch.zeros(embedding_dim))

    def reset_parameters(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.mask_vector, mean=0.0, std=0.02)

    def forward(self, rel_time):
        """
        Args:
            rel_time (Tensor): Tensor of shape [B, K] containing time values in seconds.
        Returns:
            Tensor: Encoded time features with shape [B, K, embedding_dim].
        """
        # Get the original batch dimensions
        B, K = rel_time.shape

        # Flatten the input from [B, K] to [B*K]
        flattened_time = rel_time.view(-1)

        # Apply positional encoding to the flattened input
        pos_encoded = self.pos_encoder(flattened_time)  # shape: [B*K, embedding_dim]

        # Apply a linear transformation
        linear_out = self.linear(pos_encoded)  # shape: [B*K, embedding_dim]
        linear_out = linear_out.view(B, K, -1)

        # create a mask: 1 where time is masked (i.e. < 0), else 0.
        mask = (rel_time < 0).unsqueeze(-1).float()
        mask_vector = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        # where mask==1, use mask_vector; else use linear_out.
        out = (1 - mask) * linear_out + mask * mask_vector
        return out


from torch_frame.nn.models import (
    ResNet,
)  # Ensure torch_frame is installed and imported correctly
from typing import Dict, Any


class NeighborTfsEncoder(nn.Module):
    """
    Encoder for neighbor TorchFrame objects.

    Processes a batch of lists of TorchFrame objects using a two-stage encoding style,
    similar to HeteroEncoder, for a single node type context.
    """

    def __init__(
        self,
        channels: int,
        node_type_map,  # Mapping from node type to index (if needed externally)
        col_names_dict,
        col_stats_dict,
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        """
        Args:
            channels (int): Output channels for the encoder.
            node_type_map: Mapping from node type to index.
            col_names_dict (dict): Dictionary mapping column types to list of column names.
            col_stats_dict (dict): Dictionary of statistics for columns.
            torch_frame_model_cls: Class for the TorchFrame model (default: ResNet).
            torch_frame_model_kwargs (dict): Keyword arguments for the model class.
            default_stype_encoder_cls_kwargs (dict): Dictionary mapping stype to a tuple of
                                                      (encoder class, kwargs) for that stype.
        """
        super(NeighborTfsEncoder, self).__init__()

        self.node_type_map = node_type_map
        self.inv_node_type_map = {idx: nt for nt, idx in node_type_map.items()}
        self.encoders = nn.ModuleDict()
        self.channels = channels

        # Initialize encoders for each node type using provided dictionaries
        for node_type, stype_dict in col_names_dict.items():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in stype_dict.keys()
                if stype in default_stype_encoder_cls_kwargs
            }
            self.encoders[node_type] = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=col_stats_dict[node_type],
                col_names_dict=stype_dict,
                stype_encoder_dict=stype_encoder_dict,
            )

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(self, batch_dict, neighbor_types):
        """
        Args:
            batch_dict (dict): A dictionary containing:
              - grouped_tfs[t_int]: A single concatenated TorchFrame of all neighbors
                                    for node type 't_int' in the batch.
              - grouped_indices[t_int]: The list of flat indices corresponding to
                                       each row in grouped_tfs[t_int].
              - flat_batch_idx (List[int]): The batch index 'i' for each flattened neighbor.
              - flat_nbr_idx (List[int]): The neighbor index 'j' for each flattened neighbor.
            neighbor_types (Tensor): A [B, K] tensor specifying the node type indices
                                     for each neighbor in the original (batch, neighbor) shape.

        This method performs a single-pass encoding for each node type by:
          1) Encoding the concatenated TorchFrame (big_tf) for that type in one shot.
          2) Scattering the resulting embeddings back to the flattened positions.
          3) Reassembling the final [B, K, channels] tensor using 'flat_batch_idx' and 'flat_nbr_idx'.

        Returns:
            Tensor: A [B, K, channels] tensor of encoded neighbor features, preserving
                    the original ordering of neighbors per sample.
        """
        grouped_tfs = batch_dict["grouped_tfs"]
        grouped_indices = batch_dict["grouped_indices"]
        flat_batch_idx = batch_dict["flat_batch_idx"]
        flat_nbr_idx = batch_dict["flat_nbr_idx"]

        B, K = neighbor_types.shape
        N = len(flat_batch_idx)  # total flattened neighbors
        device = neighbor_types.device

        # Pre-allocate an [N, channels] buffer
        # (Even if N==0, this works fine: shape is [0, channels].)
        encoded_flat_tensor = torch.zeros((N, self.channels), device=device)

        # 1) Encode in one shot per node type
        for t_int, big_tf in grouped_tfs.items():
            node_type_str = self.inv_node_type_map[t_int]
            encoder = self.encoders[node_type_str]

            big_tf = big_tf.to(device=device)

            for stype, tensor in big_tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    big_tf.feat_dict[stype] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )

            # assert torch.isfinite(big_tf.feat_dict[torch_frame.numerical]).all(), f"NaN/Inf in the raw big_tf for {node_type_str}?"

            out_t = encoder(
                big_tf
            )  # shape: [num_rows, channels] or [num_rows, 1, channels]
            if out_t.dim() == 3 and out_t.shape[1] == 1:
                out_t = out_t.squeeze(1)  # => [num_rows, channels]

            # Insert each row into encoded_flat_tensor
            idx_list = grouped_indices[t_int]
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            encoded_flat_tensor[idx_tensor] = out_t

        # 2) Scatter [N, channels] -> [B, K, channels]
        output = torch.zeros((B, K, self.channels), device=device)

        indices_i = torch.tensor(flat_batch_idx, dtype=torch.long, device=device)
        indices_j = torch.tensor(flat_nbr_idx, dtype=torch.long, device=device)
        output[indices_i, indices_j] = encoded_flat_tensor

        return output


from torch_geometric.nn import GINConv


class GNNPEEncoder(nn.Module):
    """
    A GNN-based positional encoder that:
      1) Assigns each node a random scalar feature from a Normal(0,1).
      2) Linearly projects it to embedding_dim.
      3) Runs a small GIN GNN on (x, edge_index, batch).
      4) Aggregates the intermediate outputs of the GNN using one of:
        - "none": use only the final layer's output,
        - "cat": concatenate all layer outputs,
        - "mean": average all layer outputs,
        - "max": max pool across all layer outputs.
      5) Returns a [B, K, embedding_dim] shaped embedding to match the rest of the pipeline.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 4,
        pooling: str = "none",
        pe_dim: int = 0,
    ):
        super().__init__()
        self.pooling = pooling.lower()
        self.num_layers = num_layers
        self.layer_embedding_dim = embedding_dim // 4
        self.pe_dim = pe_dim

        if self.pe_dim > 0:
            self.input_proj = nn.Linear(self.pe_dim, self.layer_embedding_dim)
        else:
            self.input_proj = nn.Linear(1, self.layer_embedding_dim)

        self.conv = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.layer_embedding_dim, self.layer_embedding_dim * 2),
                nn.BatchNorm1d(self.layer_embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.layer_embedding_dim * 2, self.layer_embedding_dim),
            )
            self.conv.append(GINConv(mlp, train_eps=True))

        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(self.layer_embedding_dim))

        if self.pooling == "cat":
            final_input_dim = self.layer_embedding_dim * num_layers
        elif self.pooling in ["none", "mean", "max"]:
            final_input_dim = self.layer_embedding_dim
        else:
            raise ValueError(
                "Invalid pooling method. Choose from 'none', 'cat', 'mean', 'max'."
            )

        self.final_transform = nn.Linear(final_input_dim, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        for conv in self.conv:
            for layer in conv.nn:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        nn.init.xavier_uniform_(self.final_transform.weight)
        if self.final_transform.bias is not None:
            nn.init.zeros_(self.final_transform.bias)

    def forward(self, edge_index, batch):
        """
        Args:
            edge_index (torch.Tensor): shape [2, E], the adjacency for the subgraph(s).
            batch (torch.Tensor): shape [total_nodes], specifying subgraph membership for each node.

        Returns:
            (torch.Tensor): shape [B, K, embedding_dim], a node-level embedding for each node
                            in the subgraph, where B is the batch size, K is the # of nodes in
                            each subgraph if each subgraph is the same size, or sum(K_i) if variable.
        """
        device = edge_index.device
        total_nodes = batch.size(0)

        if self.pe_dim > 0:
            data = Data(edge_index=edge_index, num_nodes=total_nodes)
            transform = T.AddLaplacianEigenvectorPE(k=self.pe_dim)
            data = transform(data)
            x_input = data.laplacian_eigenvector_pe.to(device)
        else:
            x_input = torch.randn(total_nodes, 1, device=device)

        x = self.input_proj(x_input)

        outputs = []
        for i, conv in enumerate(self.conv):
            x_res = x
            x_new = conv(x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x_res
            outputs.append(x)

        if self.pooling == "none":
            x_final = outputs[-1]
        elif self.pooling == "cat":
            x_final = torch.cat(outputs, dim=-1)
        elif self.pooling == "mean":
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.mean(outputs_tensor, dim=-1)
        elif self.pooling == "max":
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.max(outputs_tensor, dim=-1)[0]

        x = self.final_transform(x_final)

        B = batch.max().item() + 1
        K = total_nodes // B
        out = x.view(B, K, -1)

        return out
