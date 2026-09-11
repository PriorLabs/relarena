# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen,
# Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey,
# Jure Leskovec
# Vendored from snap-stanford/relgt @ 19e423ca; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGT neighbor-sampling algorithm — the tokenization core.

STATUS: the four sampler functions + two module globals are vendored verbatim
(semantically unchanged); only the import block is trimmed.
  * Source: snap-stanford/relgt @ commit 19e423ca3e7cac761130aba790857f2dc3a46ef7
    https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/utils.py
  * Vendored verbatim (no logic change): ``GLOBAL_ADJ`` / ``GLOBAL_ALL_NODES``,
    ``build_adjacency_hetero``, ``init_worker_globals``,
    ``gather_1_and_2_hop_with_seed_time``, ``_process_one_seed``.
  * Only edits vs upstream: the import block is pared to what these functions need
    (upstream's ``utils.py`` also held the dataset class / Glove embedder), and ruff
    normalizes formatting (whitespace/quotes/wrapping) — so the diff is imports +
    formatting only. No sampler logic is touched.

These functions ARE the method's local-context sampler: per seed, gather 1-hop/2-hop
neighbors under the ``time <= seed_time`` filter, select K-1 (with the random-fallback
path for neighbor-poor seeds), and build the induced subgraph. Ruff-exempt so the
sampling logic stays a clean diff against upstream. The relarena glue that drives these
over a censored graph + packs model batches lives in ``tokenize.py``, NOT here. To
re-sync: re-copy these functions from the pinned file and bump the commit.

GLOBAL_ADJ / GLOBAL_ALL_NODES are module globals the sampler reads; the driver sets
them via ``init_worker_globals`` before iterating (serial) or per worker (Pool).
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch_geometric.data import HeteroData

GLOBAL_ADJ = None
GLOBAL_ALL_NODES = None


def build_adjacency_hetero(hetero_data: HeteroData, undirected: bool = True):
    adjacency = {
        node_type: [set() for _ in range(hetero_data[node_type].num_nodes)]
        for node_type in hetero_data.node_types
    }
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if "edge_index" not in hetero_data[edge_type]:
            continue
        edge_index = hetero_data[edge_type].edge_index
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        for s, d in zip(src_list, dst_list):
            adjacency[src_type][s].add((dst_type, d))
            if undirected:
                adjacency[dst_type][d].add((src_type, s))
    return adjacency


def init_worker_globals(adj, all_nodes):
    global GLOBAL_ADJ, GLOBAL_ALL_NODES
    GLOBAL_ADJ = adj
    GLOBAL_ALL_NODES = all_nodes


def gather_1_and_2_hop_with_seed_time(
    adjacency: Dict[str, List[set]],
    data: HeteroData,
    node_type: str,
    node_idx: int,
    seed_time: float,
    max_1hop_threshold: int = 5000,
    max_2hop_threshold: int = 1000,
) -> List[Tuple[str, int, int, float, Optional[set]]]:
    """
    Gather 1-hop and 2-hop neighbors with time condition.

    Returns:
        neighbors_with_time: List of tuples:
            (nbr_t, nbr_i, hop, relative_time_days, None) for 1-hop
            (nbr_t, nbr_i, hop, relative_time_days, connecting_1hop_tuple) for 2-hop
    """
    # Gather 1-hop neighbors satisfying the time condition
    n1_full = adjacency[node_type][node_idx]
    if len(n1_full) > max_1hop_threshold:
        n1_full = random.sample(list(n1_full), max_1hop_threshold)
    else:
        n1_full = list(n1_full)

    n1 = set()
    for nbr_t, nbr_i in n1_full:
        if hasattr(data[nbr_t], "time"):
            if data[nbr_t].time[nbr_i] <= seed_time:
                n1.add((nbr_t, nbr_i))
        else:
            n1.add((nbr_t, nbr_i))

    # Gather 2-hop neighbors satisfying the time condition
    n2 = defaultdict(set)  # Map 2-hop neighbor to set of connecting 1-hop neighbors
    for nbr_t, nbr_i in n1:
        nbr2_full = adjacency[nbr_t][nbr_i]
        if len(nbr2_full) > max_2hop_threshold:
            nbr2_full = random.sample(list(nbr2_full), max_2hop_threshold)
        else:
            nbr2_full = list(nbr2_full)

        for nbr2_t, nbr2_i in nbr2_full:
            # Skip if we loop back to the original node
            if (nbr2_t, nbr2_i) == (node_type, node_idx):
                continue
            if hasattr(data[nbr2_t], "time"):
                if data[nbr2_t].time[nbr2_i] <= seed_time:
                    n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))
            else:
                n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))

    # Remove overlaps: ensure 2-hop neighbors are not already in 1-hop
    n2 = {k: v for k, v in n2.items() if k not in n1}

    neighbors_with_time = []

    # Process 1-hop neighbors with hop distance 1
    for nbr_t, nbr_i in n1:
        if hasattr(data[nbr_t], "time"):
            nbr_time = data[nbr_t].time[nbr_i].item()
            relative_time_days = (seed_time - nbr_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0  # no time entities
        # Append tuple with hop level 1 and no connecting 1-hop neighbor
        neighbors_with_time.append((nbr_t, nbr_i, 1, relative_time_days, None))

    # Process 2-hop neighbors with hop distance 2
    for (nbr2_t, nbr2_i), connecting_1hops in n2.items():
        if hasattr(data[nbr2_t], "time"):
            nbr2_time = data[nbr2_t].time[nbr2_i].item()
            relative_time_days = (seed_time - nbr2_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0  # no time entities
        # If multiple connecting 1-hop neighbors, we will handle in sampling
        neighbors_with_time.append(
            (nbr2_t, nbr2_i, 2, relative_time_days, connecting_1hops)
        )

    return neighbors_with_time


def _process_one_seed(args):
    """
    Worker function: gather neighbors for a single seed node + time,
    perform local nodes expansions up to K, apply fallback if necessary,
    then return a final list of neighbor tokens.
    """
    global GLOBAL_ADJ, GLOBAL_ALL_NODES

    (data, K, seed_node_type, seed_node_idx, seed_time, seed_val) = args
    random.seed(seed_val)

    # 1. gather 1-hop and 2-hop
    T_hat = gather_1_and_2_hop_with_seed_time(
        GLOBAL_ADJ, data, seed_node_type, seed_node_idx, seed_time
    )
    T_hat_list = list(T_hat)
    size_th = len(T_hat_list)
    K_minus_1 = K - 1

    # separate 1-hop, 2-hop
    one_hop_neighbors = [n for n in T_hat_list if n[2] == 1]
    two_hop_neighbors = [n for n in T_hat_list if n[2] == 2]
    combined_neighbors = one_hop_neighbors + two_hop_neighbors

    # 2. If we have enough neighbors => random.sample
    #    If not => random.choices
    if size_th >= K_minus_1:
        chosen_neighbors = random.sample(combined_neighbors, K_minus_1)
    elif 0 < size_th < K_minus_1:
        chosen_neighbors = random.choices(combined_neighbors, k=K_minus_1)
    else:
        # fallback from GLOBAL_ALL_NODES
        if K_minus_1 <= len(GLOBAL_ALL_NODES):
            fallback = random.sample(GLOBAL_ALL_NODES, K_minus_1)
        else:
            fallback = random.choices(GLOBAL_ALL_NODES, k=K_minus_1)
        chosen_neighbors = []
        for ft, fi in fallback:
            if hasattr(data[ft], "time"):
                ft_time = data[ft].time[fi].item()
                rel_time = (seed_time - ft_time) / (60 * 60 * 24)
            else:
                rel_time = 0
            chosen_neighbors.append((ft, fi, 3, rel_time, None))

    # 3. Build final_tokens with subgraph adj for the seed node and its chosen neighbors
    final_tokens = []
    seed_token = (seed_node_type, seed_node_idx, 0, 0.0, 0)
    final_tokens.append(seed_token)
    final_tokens.extend(chosen_neighbors)

    # randomize order except keep seed first
    if len(final_tokens) > 1:
        first = final_tokens[0]
        rest = final_tokens[1:]
        rest = random.sample(rest, len(rest))
        final_tokens = [first] + rest

    # build adjacency among these K nodes
    local_map = {}
    for j, (t_str, i, hop, t_val, c1hops) in enumerate(final_tokens):
        local_map[(t_str, i)] = j

    edges = []
    for j_src, (t_str, i, hop, t_val, c1hops) in enumerate(final_tokens):
        for nbr_t, nbr_i in GLOBAL_ADJ[t_str][i]:
            if (nbr_t, nbr_i) in local_map:
                j_dst = local_map[(nbr_t, nbr_i)]
                edges.append((j_src, j_dst))

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    else:
        arr = np.array(edges, dtype=np.int32)
        edge_index = arr.T  # shape [2, E]

    return (seed_node_type, seed_node_idx, final_tokens, edge_index)
