"""relarena glue around the vendored RelGT sampler.

STATUS: NEW relarena code — NOT vendored. (The sampling *algorithm* it calls lives
verbatim in `_sampler.py`; this file is the I/O plumbing around it.) It re-implements
the role of upstream's `RelGTTokens` (`utils.py`) — which seeds to sample, where to
store the result, and how to pack the model batch — written fresh to fit relarena's
fit/predict contract. The batch `collate` produces is kept STRUCTURALLY IDENTICAL to
upstream's `RelGTTokens.collate` so the vendored model is fed exactly what it expects.

Like upstream, tokens are sampled once and cached to HDF5 (same datasets: `types` /
`indices` / `hops` / `times` + a CSR-style `edges` / `edges_offsets`), then
reused every epoch. The deviations are I/O only; none change which neighbors are drawn:

  * Graph source: samples on the **censored** `HeteroData` relarena hands `fit` /
    `predict` (per-split cutoff), so the leakage guarantee is the harness's, not
    upstream's single test-censored graph.
  * Cache key: a content hash of the graph's censoring fingerprint (per-type node count
    + max node time) + the split's seeds + `K` — NOT upstream's `{dataset}/{task}/
    {split}` path. That path collides under relarena (the same train seeds are sampled
    on the val-censored *and* test-censored graphs); fingerprinting the graph keeps the
    inner (tuning) and outer (refit) token sets in separate cache entries.
  * Adjacency is rebuilt per call (never cached in the module globals across graphs),
    so a re-fit on a differently censored DB can't reuse a stale adjacency.
  * Global node ids use a closed-form cumulative offset instead of upstream's
    `(type, local) -> global` dict — identical ids, but no giant dict on big graphs.
  * An explicit `CacheConfig` selects private scratch, fill, or read-only behavior;
    completed HDF5 files are loaded fully into RAM before use or publication.
  * Default execution is serial; `num_workers > 1` fans the verbatim worker out.
  * Drops upstream's per-item `sample["tfs"]` field — `collate` rebuilds the grouped
    TorchFrames itself and the model never reads it, so it was dead per-item work.
"""

from __future__ import annotations

import gc
import hashlib
import multiprocessing as mp
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np
import torch
from relbench.base import EntityTask, Table
from relbench.modeling.graph import get_node_train_table_input
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from relarena.cache import CacheConfig, cache_key, cached_artifact
from relarena.identity import RunIdentity
from relarena.models.relgt._vendor._sampler import (
    _process_one_seed,
    build_adjacency_hetero,
    init_worker_globals,
)

#: Hop label of the random-fallback tokens; one past the 2-hop sampler, so the type/hop
#: embeddings are sized for hops {0 (seed), 1, 2, 3 (fallback)}.
MAX_NEIGHBOR_HOP = 3

#: Bumped when token sampling changes so an old HDF5 entry is never reused.
_TOKEN_CACHE_VERSION = 3


def _type_offsets(data: HeteroData) -> dict[str, int]:
    """Cumulative node-count offset per type, in `data.node_types` order.

    `offset[type] + local_idx` is the global node id over the concatenated all-types
    index space the codebook (`c_idx` buffer, sized to the total node count) expects —
    the closed form of upstream's `_create_global_mappings` dict.
    """
    offsets: dict[str, int] = {}
    running = 0
    for node_type in data.node_types:
        offsets[node_type] = running
        running += data[node_type].num_nodes
    return offsets


def _graph_fingerprint(data: HeteroData) -> tuple:
    """Per-type `(name, num_nodes, max_time)` — captures the censoring of `data`.

    The val-censored and test-censored graphs of the same task differ in node count
    and/or max node time, so this distinguishes the inner and outer cache entries.
    """
    fp = []
    for nt in data.node_types:
        store = data[nt]
        max_time = int(store.time.max()) if hasattr(store, "time") else -1
        fp.append((nt, int(store.num_nodes), max_time))
    return tuple(fp)


def token_cache_key(
    data: HeteroData,
    task: EntityTask,
    node_idxs: torch.Tensor,
    times: torch.Tensor,
    k: int,
    undirected: bool,
    run_identity: RunIdentity | None,
) -> PurePosixPath:
    """Return RelGT's complete token key from every sampling dependency."""
    if run_identity is None:
        dataset, dataset_fingerprint, phase = "direct", "direct", "direct"
        task_name = f"{task.entity_table}-{task.target_col}"
        task_fingerprint = _digest(
            repr(
                (
                    task.entity_table,
                    task.entity_col,
                    task.time_col,
                    task.target_col,
                )
            ).encode()
        )
    else:
        dataset = run_identity.dataset
        dataset_fingerprint = run_identity.dataset_fingerprint or "unversioned"
        phase = run_identity.phase or "direct"
        task_name = run_identity.task or f"{task.entity_table}-{task.target_col}"
        task_fingerprint = run_identity.task_fingerprint or "unversioned"
    seeds = hashlib.blake2s(digest_size=8)
    seeds.update(node_idxs.cpu().numpy().astype(np.int64).tobytes())
    seeds.update(times.cpu().numpy().astype(np.int64).tobytes())
    segments = [
        "relgt",
        f"tokens-v{_TOKEN_CACHE_VERSION}",
        f"{dataset}@{dataset_fingerprint}",
        f"{task_name}@{task_fingerprint}",
        f"phase-{phase}",
    ]
    if run_identity is not None and run_identity.data_version is not None:
        segments.append(f"data-{_digest(run_identity.data_version.encode())}")
    segments.extend(
        (
            f"graph-{_digest(repr(_graph_fingerprint(data)).encode())}",
            f"seeds-{seeds.hexdigest()}",
            f"k-{k}-undirected-{int(undirected)}.h5",
        )
    )
    return cache_key(*segments)


def _digest(value: bytes) -> str:
    return hashlib.blake2s(value, digest_size=8).hexdigest()


def _require_persistent_identity(
    cache: CacheConfig, run_identity: RunIdentity | None
) -> None:
    if (
        cache.directory is not None
        and cache.on_miss != "compute"
        and (
            run_identity is None
            or run_identity.dataset_fingerprint is None
            or run_identity.task_fingerprint is None
        )
    ):
        raise ValueError(
            "Persistent RelGT tokens need dataset and task fingerprints in "
            "RunIdentity; use a benchmark/PQ entrypoint or pass one explicitly."
        )


def _sampling_seed(node_type: str, node_idx: int, seed_time: float, k: int) -> int:
    """Return a process-stable 32-bit seed for one sampled context."""
    value = repr((node_type, node_idx, seed_time, k)).encode()
    return int.from_bytes(hashlib.blake2s(value, digest_size=4).digest())


#: The graph shared with sampler workers. Set in the parent before the `fork` Pool
#: so workers inherit it copy-on-write instead of pickling the (huge) graph per task —
#: the upstream parallel path puts `data` in every task tuple, which blows up on the
#: large datasets. Read only by `_sample_one` in the worker.
_WORKER_DATA: HeteroData | None = None


def _sample_one(task: tuple) -> tuple:
    """Worker entry: prepend the fork-inherited graph, then call the verbatim sampler.

    `task` is the per-seed fields without the graph
    (`k, node_type, node_idx, seed_time, seed_val`); `_process_one_seed` expects the
    graph first, so it is spliced back in from the shared global here.
    """
    return _process_one_seed((_WORKER_DATA, *task))


def _tokens_to_arrays(
    results: list[tuple[list[tuple], np.ndarray]],
    k: int,
    node_type_to_index: dict[str, int],
) -> dict[str, Any]:
    """Flatten per-seed token lists into dense `[N, K]` arrays + ragged edge lists."""
    n = len(results)
    types = np.zeros((n, k), dtype=np.int16)
    indices = np.zeros((n, k), dtype=np.int32)
    hops = np.zeros((n, k), dtype=np.int8)
    times = np.zeros((n, k), dtype=np.float32)
    edges: list[np.ndarray] = [None] * n
    for i, (tokens, edge_index) in enumerate(results):
        for j, (t_str, nbr_idx, hop, rel_time, _) in enumerate(tokens):
            types[i, j] = node_type_to_index[t_str]
            indices[i, j] = nbr_idx
            hops[i, j] = hop
            times[i, j] = rel_time
        edges[i] = edge_index
    return {
        "types": types,
        "indices": indices,
        "hops": hops,
        "times": times,
        "edges": edges,
    }


def _read_cache(path: Path) -> dict[str, Any]:
    """Load a token cache fully into memory (NFS-robust: one read, no held handle).

    Reading lazily per item from an HDF5 file on NFS is fragile — a handle held across
    a rename goes stale (ESTALE) — so we slurp the whole file once and serve from RAM.
    """
    with h5py.File(path, "r", locking=False) as hf:
        arrays = {key: hf[key][:] for key in ("types", "indices", "hops", "times")}
        offsets = hf["edges_offsets"][:]
        flat = hf["edges"][:]
    edges = [
        flat[:, int(offsets[i]) : int(offsets[i + 1])]
        for i in range(arrays["types"].shape[0])
    ]
    arrays["edges"] = edges
    return arrays


def _validate_arrays(
    arrays: dict[str, Any], n: int, k: int, num_node_types: int
) -> None:
    """Validate the HDF5 payload before tokens reach the model or publication."""
    for name in ("types", "indices", "hops", "times"):
        if arrays[name].shape != (n, k):
            raise ValueError(
                f"corrupt RelGT tokens: {name} has shape {arrays[name].shape}, "
                f"expected {(n, k)}"
            )
    if len(arrays["edges"]) != n:
        raise ValueError(
            f"corrupt RelGT tokens: {len(arrays['edges'])} edge lists, expected {n}"
        )
    if arrays["types"].size and (
        arrays["types"].min() < 0 or arrays["types"].max() >= num_node_types
    ):
        raise ValueError("corrupt RelGT tokens: node type index out of range")
    if arrays["hops"].size and (
        arrays["hops"].min() < 0 or arrays["hops"].max() > MAX_NEIGHBOR_HOP
    ):
        raise ValueError("corrupt RelGT tokens: hop index out of range")


#: Seeds processed per chunk when writing the cache. Bounds peak memory to one
#: chunk's tokens (not all N seeds), matching upstream's chunked-incremental write —
#: the all-at-once path blows past 300 GB on multi-million-seed tables (rel-hm).
_WRITE_CHUNK = 100_000


def _sample_write_chunked(
    path: Path,
    data: HeteroData,
    k: int,
    node_type: str,
    node_idxs: torch.Tensor,
    times: torch.Tensor,
    *,
    undirected: bool,
    num_workers: int,
    node_type_to_index: dict[str, int],
) -> None:
    """Sample + write tokens to `path` in seed-chunks (bounded memory).

    Upstream's scheme: sample `_WRITE_CHUNK` seeds, write the dense `[chunk, K]`
    arrays + their edges to the HDF5, free, repeat — so peak memory is one chunk rather
    than every seed's tokens at once (which OOMs / drags on tables like rel-hm
    item-sales at ~5.6 M seeds). Output is identical to the all-at-once path.
    """
    global _WORKER_DATA
    adjacency = build_adjacency_hetero(data, undirected=undirected)
    all_nodes = [(nt, i) for nt in data.node_types for i in range(data[nt].num_nodes)]
    init_worker_globals(adjacency, all_nodes)  # parent globals; workers inherit on fork
    _WORKER_DATA = data

    n = int(node_idxs.numel())
    idxs = node_idxs.tolist()
    tlist = [float(t) for t in times.tolist()]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move the adjacency (hundreds of millions of small Python objects on large
    # graphs) to a permanent GC generation before forking workers: the collector
    # never scans frozen objects, so their fork-COW pages stay shared instead of
    # being copied into every worker on the next GC pass (the rel-hm OOM).
    gc.collect()
    gc.freeze()
    pool = (
        mp.get_context("fork").Pool(processes=num_workers) if num_workers > 1 else None
    )
    try:
        with h5py.File(path, "w", locking=False) as hf:
            d_types = hf.create_dataset("types", shape=(n, k), dtype=np.int16)
            d_indices = hf.create_dataset("indices", shape=(n, k), dtype=np.int32)
            d_hops = hf.create_dataset("hops", shape=(n, k), dtype=np.int8)
            d_times = hf.create_dataset("times", shape=(n, k), dtype=np.float32)
            d_edges = hf.create_dataset(
                "edges",
                shape=(2, 0),
                maxshape=(2, None),
                dtype="int32",
                chunks=(2, 1 << 20),
            )
            offsets = np.zeros(n + 1, dtype=np.uint64)
            edge_total = 0
            for start in range(0, n, _WRITE_CHUNK):
                end = min(start + _WRITE_CHUNK, n)
                tasks = [
                    (
                        k,
                        node_type,
                        idxs[i],
                        tlist[i],
                        _sampling_seed(node_type, idxs[i], tlist[i], k),
                    )
                    for i in range(start, end)
                ]
                if pool is not None:
                    cs = max(1, len(tasks) // (num_workers * 8))
                    res = pool.map(_sample_one, tasks, chunksize=cs)
                else:
                    res = [_sample_one(t) for t in tasks]
                results = [(tok, edge) for (_, _, tok, edge) in res]
                arr = _tokens_to_arrays(results, k, node_type_to_index)
                d_types[start:end] = arr["types"]
                d_indices[start:end] = arr["indices"]
                d_hops[start:end] = arr["hops"]
                d_times[start:end] = arr["times"]
                blocks = []
                for j, edge in enumerate(arr["edges"]):
                    w = 0 if edge is None else edge.shape[1]
                    offsets[start + j + 1] = offsets[start + j] + w
                    if w:
                        blocks.append(edge)
                if blocks:
                    block = np.concatenate(blocks, axis=1)
                    d_edges.resize((2, edge_total + block.shape[1]))
                    d_edges[:, edge_total : edge_total + block.shape[1]] = block
                    edge_total += int(block.shape[1])
                del res, results, arr, blocks
                gc.collect()
            hf.create_dataset("edges_offsets", data=offsets)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


class RelGTTokens(Dataset):
    """Per-seed RelGT token sequences for one split, sampled on the censored graph.

    Each item is the seed's K-token context (seed at index 0) plus its induced
    subgraph; `collate` packs a batch into the kwargs `RelGT.forward` takes.
    Tokens live in memory for serving; the explicit cache policy controls whether
    their HDF5 artifact is private scratch, filled, or loaded read-only.
    """

    def __init__(
        self,
        data: HeteroData,
        task: EntityTask,
        table: Table,
        k: int,
        *,
        cache: CacheConfig | None = None,
        run_identity: RunIdentity | None = None,
        undirected: bool = True,
        num_workers: int = 0,
    ) -> None:
        """Sample (or load cached) tokens for every seed in `table` against `data`.

        `table` is the split's label table (masked on the test split, so `target` is
        then `None`); `data` must be censored at that split's cutoff.
        """
        super().__init__()
        self.data = data
        self.k = k

        table_input = get_node_train_table_input(table=table, task=task)
        self.node_type, node_idxs = table_input.nodes
        self.target = table_input.target
        if table_input.time is None:
            raise ValueError("RelGT requires seed timestamps; task has no time column.")

        self.node_types = data.node_types
        self.node_type_to_index = {nt: i for i, nt in enumerate(self.node_types)}
        self.index_to_node_type = dict(enumerate(self.node_types))
        self._offsets = _type_offsets(data)
        self.num_global_nodes = sum(data[nt].num_nodes for nt in self.node_types)
        self._n = node_idxs.numel()

        cache = cache or CacheConfig(None, "compute")
        _require_persistent_identity(cache, run_identity)
        key = token_cache_key(
            data,
            task,
            node_idxs,
            table_input.time,
            k,
            undirected,
            run_identity,
        )

        def build(path: Path) -> dict[str, Any]:
            _sample_write_chunked(
                path,
                data,
                k,
                self.node_type,
                node_idxs,
                table_input.time,
                undirected=undirected,
                num_workers=num_workers,
                node_type_to_index=self.node_type_to_index,
            )
            return _read_cache(path)

        def validate(arrays: dict[str, Any]) -> None:
            _validate_arrays(arrays, self._n, k, len(self.node_types))

        self._mem = cached_artifact(
            cache,
            key,
            storage="file",
            load=_read_cache,
            build=build,
            validate=validate,
            warm_hint="Run python -m relarena.models.relgt.warm_cache.",
        )

    @property
    def node_type_map(self) -> dict[str, int]:
        """Node-type → integer id map the model's type embedding is sized from."""
        return self.node_type_to_index

    def __len__(self) -> int:
        """Number of seed entities in the split."""
        return self._n

    def _global_index(self, type_idx: int, local_idx: int) -> int:
        """Global node id over the concatenated all-types index space."""
        return self._offsets[self.index_to_node_type[type_idx]] + local_idx

    def _row(self, idx: int) -> tuple[np.ndarray, ...]:
        """Return `(types, indices, hops, times, edge_index)` for one seed."""
        m = self._mem
        return (
            m["types"][idx],
            m["indices"][idx],
            m["hops"][idx],
            m["times"][idx],
            m["edges"][idx],
        )

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], torch.Tensor | None]:
        """Return `(sample, label)` for one seed (`label` is `None` on test)."""
        types, indices, hops, times, edge_np = self._row(idx)
        edge_index = (
            torch.zeros((2, 0), dtype=torch.long)
            if edge_np is None or edge_np.size == 0
            else torch.from_numpy(np.asarray(edge_np)).long()
        )
        sample = {
            "types": torch.from_numpy(types.astype(np.int64)),
            "indices": torch.from_numpy(indices.astype(np.int64)),
            "hops": torch.from_numpy(hops.astype(np.int64)),
            "times": torch.from_numpy(times.astype(np.float32)),
            "edge_index": edge_index,
            "first_type": int(types[0]),
            "first_index": int(indices[0]),
            "global_idx": idx,
        }
        label = self.target[idx] if self.target is not None else None
        return sample, label

    def collate(self, batch: list[tuple[dict[str, Any], torch.Tensor | None]]) -> dict:
        """Pack a batch into the `RelGT.forward` kwargs (see module docstring)."""
        samples, labels = zip(*batch)

        neighbor_types = torch.stack([s["types"] for s in samples], dim=0)  # [B, K]
        neighbor_indices = torch.stack([s["indices"] for s in samples], dim=0)
        neighbor_hops = torch.stack([s["hops"] for s in samples], dim=0)
        neighbor_times = torch.stack([s["times"] for s in samples], dim=0)
        b, k = neighbor_types.shape

        node_indices = torch.tensor(
            [self._global_index(s["first_type"], s["first_index"]) for s in samples],
            dtype=torch.long,
        )

        # Group every neighbor of the same node type across the batch into one
        # TorchFrame so each type's encoder runs once; the flat positions scatter the
        # encoded rows back to [B, K, channels].
        grouped_tfs: dict[int, Any] = {}
        grouped_indices: dict[int, list[int]] = {}
        for t_id in range(len(self.node_types)):
            mask = neighbor_types == t_id
            if not mask.any():
                continue
            local_idxs = neighbor_indices[mask]
            type_str = self.index_to_node_type[t_id]
            positions = [int(bi) * k + int(ki) for bi, ki in mask.nonzero().tolist()]
            grouped_tfs[t_id] = self.data[type_str].tf[local_idxs]
            grouped_indices[t_id] = positions

        flat_batch_idx = torch.arange(b).unsqueeze(1).expand(b, k).reshape(-1).tolist()
        flat_nbr_idx = torch.arange(k).repeat(b).tolist()

        # Batch the per-sample induced subgraphs with a running node offset (PyG style).
        batched_edges, batch_vec, offset = [], [], 0
        for i, s in enumerate(samples):
            batched_edges.append(s["edge_index"] + offset)
            batch_vec.append(torch.full((s["types"].size(0),), i, dtype=torch.long))
            offset += s["types"].size(0)
        edge_index = torch.cat(batched_edges, dim=1)
        batch = torch.cat(batch_vec, dim=0)

        return {
            "neighbor_types": neighbor_types,
            "neighbor_indices": neighbor_indices,
            "neighbor_hops": neighbor_hops,
            "neighbor_times": neighbor_times,
            "labels": None if self.target is None else torch.stack(labels, dim=0),
            "node_indices": node_indices,
            "grouped_tfs": grouped_tfs,
            "grouped_indices": grouped_indices,
            "flat_batch_idx": flat_batch_idx,
            "flat_nbr_idx": flat_nbr_idx,
            "global_idx": torch.tensor(
                [s["global_idx"] for s in samples], dtype=torch.long
            ),
            "edge_index": edge_index,
            "batch": batch,
        }


def precompute_tokens(
    data: HeteroData,
    task: EntityTask,
    tables: list[Table],
    k: int,
    cache: CacheConfig,
    *,
    run_identity: RunIdentity | None = None,
    undirected: bool = True,
    num_workers: int = 0,
) -> None:
    """Populate the on-disk token cache for each of `tables` against `data`.

    A warm job builds each split's censored graph once and calls this for that split's
    tables, using the same key factory and builder as fit/predict.
    """
    for table in tables:
        RelGTTokens(
            data,
            task,
            table,
            k,
            cache=cache,
            run_identity=run_identity,
            undirected=undirected,
            num_workers=num_workers,
        )
