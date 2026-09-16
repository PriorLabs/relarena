"""Shared internals of the GNN baselines (`graphsage`, `relgnn`, `relgt`).

The three differ in their message passing, not in how the graph is materialized
or how an epoch is stepped: `graph` builds the PyG hetero graph and memoizes it,
`training` holds the task setup / train / infer loop, and `graph_cache` is the
dependency-free memo they use. `_vendor` holds a faithful copy of RelBench's
example GNN building blocks, which are not importable from the installed
package; see `NOTICE` and `models/VENDORED-LICENSES`.
"""
