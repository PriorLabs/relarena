# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen,
# Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey,
# Jure Leskovec
# Vendored from snap-stanford/relgt @ 19e423ca; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGT model — the EMA vector-quantized codebook for global attention (``VectorQuantizerEMA``).

STATUS: vendored 1:1 — NO modifications.
  * Source: snap-stanford/relgt @ commit 19e423ca3e7cac761130aba790857f2dc3a46ef7
    https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/codebook.py
  * Semantically verbatim — no code logic changed. Vs upstream: the leading
    ``# ruff: noqa`` + this docstring are added and ruff normalizes formatting
    (whitespace/quotes/wrapping), so the diff is formatting-only. The unused
    ``from re import X`` is kept because it is present upstream.

Ruff-exempt so it stays a clean diff against upstream. relarena glue (fit/predict, the
tokenization adapter, the search space, caches) is NOT here — it lives in
``relarena.models.relgt`` and ``relarena.models.relgt.tokenize``. To re-sync: re-copy
from the pinned file and bump the commit.
"""

from re import X
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) for the codebook.
    Adapted from https://github.com/devnkong/GOAT

    Args:
        num_embeddings (int): The number of embeddings in the codebook.
        embedding_dim (int): The dimensionality of each embedding.
        decay (float, optional): The decay rate for the EMA. Defaults to 0.99.

    Attributes:
        _embedding_dim (int): The dimensionality of each embedding.
        _num_embeddings (int): The number of embeddings in the codebook.
        _decay (float): The decay rate for the EMA.
        _embedding (nn.Embedding): The embedding matrix.
        _ema_cluster_size (torch.Tensor): The exponential moving average of the cluster sizes.
        _ema_w (torch.Tensor): The exponential moving average of the embedding updates.
    """

    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer(
            "_embedding", torch.randn(self._num_embeddings, self._embedding_dim)
        )
        self.register_buffer(
            "_embedding_output",
            torch.randn(self._num_embeddings, self._embedding_dim),
        )
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", torch.randn(self._num_embeddings, self._embedding_dim)
        )

        self._decay = decay
        self.bn = torch.nn.BatchNorm1d(self._embedding_dim, affine=False)

    def reset_parameters(self):
        nn.init.normal_(self._embedding, mean=0.0, std=1.0)
        nn.init.normal_(self._embedding_output, mean=0.0, std=1.0)
        nn.init.zeros_(self._ema_cluster_size)
        nn.init.normal_(self._ema_w, mean=0.0, std=1.0)

        self.bn.reset_parameters()

    def get_k(self):
        """
        Returns the key tensor of the embedding matrix.
        """
        return self._embedding_output

    def get_v(self):
        """
        Returns the value tensor of the embedding matrix.
        """
        return self._embedding_output[:, : self._embedding_dim]

    def update(self, x):
        inputs_normalized = self.bn(x)
        embedding_normalized = self._embedding

        # Calculate distances
        distances = (
            torch.sum(inputs_normalized**2, dim=1, keepdim=True)
            + torch.sum(embedding_normalized**2, dim=1)
            - 2 * torch.matmul(inputs_normalized, embedding_normalized.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (
                (self._ema_cluster_size + 1e-5) / (n + self._num_embeddings * 1e-5) * n
            )

            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

            running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(dim=0)
            running_mean = self.bn.running_mean.unsqueeze(dim=0)
            self._embedding_output.data = self._embedding * running_std + running_mean

        return encoding_indices
