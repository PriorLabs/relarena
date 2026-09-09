# ruff: noqa
# SPDX-License-Identifier: MIT
# Copyright (c) 2023 RelBench Team
# Vendored from snap-stanford/RelGNN @ cffdb8b; full license text in
# models/VENDORED-LICENSES.
"""Vendored RelGNN atomic-routes builder (``get_atomic_routes``).

STATUS: vendored from upstream; kept verbatim (no semantic change).
  * Source: snap-stanford/RelGNN @ commit cffdb8b54627e92c7dd112c1243dde739c90d35b
    https://github.com/snap-stanford/RelGNN/blob/cffdb8b54627e92c7dd112c1243dde739c90d35b/examples/atomic_routes.py

Atomic routes are the simple source->destination paths RelGNN message-passes over:
``dim-dim`` (a single fkey edge + its reverse) and ``dim-fact-dim`` (two fkeys sharing
a fact table, composed into a direct route between the two dimension tables). Pure graph
topology — derived from the edge-type schema, no data or hyperparameters.

Ruff-exempt (``# ruff: noqa``) so it stays a clean diff against upstream. To re-sync:
re-copy from the pinned file and bump the commit above.
"""

from collections import defaultdict


def get_atomic_routes(edge_type_list):

    src_to_tuples = defaultdict(list)
    for src, rel, dst in edge_type_list:
        if rel.startswith("f2p"):
            if src == dst:
                src = src + "--" + rel
            src_to_tuples[src].append((src, rel, dst))

    atomic_routes_list = []
    get_rev_edge = lambda edge: (edge[2], "rev_" + edge[1], edge[0])
    for src, tuples in src_to_tuples.items():
        if "--" in src:
            src = src.split("--")[0]
        if len(tuples) == 1:
            _, rel, dst = tuples[0]
            edge = (src, rel, dst)
            atomic_routes_list.append(("dim-dim",) + edge)
            atomic_routes_list.append(("dim-dim",) + get_rev_edge(edge))
        else:
            for _, rel_q, dst_q in tuples:
                for _, rel_v, dst_v in tuples:
                    if rel_q != rel_v:
                        edge_q = (src, rel_q, dst_q)
                        edge_v = (src, rel_v, dst_v)
                        atomic_routes_list.append(
                            ("dim-fact-dim",) + edge_q + get_rev_edge(edge_v)
                        )

    return atomic_routes_list
