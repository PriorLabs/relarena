"""Tests for the RelGT tokenization glue (`models/relgt/tokenize.py`).

Drive the real `RelGTTokens` (verbatim sampler + collate) on a small synthetic
heterogeneous temporal graph — no RelBench download — and assert the batch contract the
vendored model relies on: seed token at index 0, the `time <= seed_time` leakage guard
on sampled neighbors, global seed-node ids, and that the packed batch actually feeds
`RelGT.forward`. The real-data path is covered by the CLI smoke run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
torch_frame = pytest.importorskip("torch_frame")
from relbench.base import Table, TaskType  # noqa: E402
from relbench.modeling.graph import get_node_train_table_input  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from relarena.cache import CacheConfig  # noqa: E402
from relarena.identity import RunIdentity  # noqa: E402
from relarena.models.relgt.tokenize import (  # noqa: E402
    RelGTTokens,
    _require_persistent_identity,
    precompute_tokens,
    token_cache_key,
)

TensorFrame = torch_frame.TensorFrame
stype = torch_frame.stype

_BASE = 1_577_836_800  # 2020-01-01 UTC, in UNIX seconds (to_unix_time's unit)
_DAY = 86_400


def _identity(phase: str = "inner") -> RunIdentity:
    return RunIdentity("dataset", "dbfp", "task", "taskfp", phase=phase)


def test__persistent_token_cache__requires_owner_specific_fingerprints(
    tmp_path: Path,
) -> None:
    incomplete = RunIdentity("dataset", None, "task", None)
    with pytest.raises(ValueError, match="Persistent RelGT tokens"):
        _require_persistent_identity(CacheConfig(tmp_path, "fill"), incomplete)

    _require_persistent_identity(CacheConfig(None, "compute"), incomplete)
    _require_persistent_identity(CacheConfig(tmp_path, "compute"), incomplete)


def _tensor_frame(num_rows: int, n_cols: int) -> "TensorFrame":
    return TensorFrame(
        feat_dict={stype.numerical: torch.randn(num_rows, n_cols)},
        col_names_dict={stype.numerical: [f"c{i}" for i in range(n_cols)]},
    )


def _synthetic_graph(n1: int = 30) -> HeteroData:
    """Two temporal node types (t0 seeds, t1) with t0<->t1 edges; times in UNIX s.

    `n1` varies the t1 node count so two calls model differently-censored graphs
    (distinct fingerprints), as the inner/outer splits would produce.
    """
    n0, n_cols = 20, 2
    data = HeteroData()
    data["t0"].num_nodes = n0
    data["t1"].num_nodes = n1
    data["t0"].time = torch.tensor([_BASE + i * _DAY for i in range(n0)])
    data["t1"].time = torch.tensor([_BASE + j * _DAY for j in range(n1)])
    data["t0"].tf = _tensor_frame(n0, n_cols)
    data["t1"].tf = _tensor_frame(n1, n_cols)
    # Each t0 connects to three t1 nodes; undirected sampling makes other t0 reachable
    # at hop 2. One edge type is enough (the sampler symmetrizes it).
    src = [i for i in range(n0) for _ in range(3)]
    dst = [(i + d) % n1 for i in range(n0) for d in range(3)]
    data["t0", "to", "t1"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    return data


def _task_and_table(seed_ids: list[int]) -> tuple[SimpleNamespace, Table]:
    """A minimal node task + label table for `get_node_train_table_input`.

    Seed cutoffs sit mid-range so the `time <= seed_time` filter actually drops some
    neighbors (a vacuous filter would pass the leakage assertion trivially).
    """
    task = SimpleNamespace(
        entity_table="t0",
        entity_col="id",
        target_col="y",
        time_col="ts",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    cutoffs = [pd.Timestamp(_BASE + (sid + 5) * _DAY, unit="s") for sid in seed_ids]
    df = pd.DataFrame(
        {"id": seed_ids, "ts": cutoffs, "y": [sid % 2 for sid in seed_ids]}
    )
    table = Table(df=df, fkey_col_to_pkey_table={}, pkey_col=None, time_col="ts")
    return task, table


@pytest.fixture
def tokens() -> RelGTTokens:
    seed_ids = list(range(8))
    task, table = _task_and_table(seed_ids)
    return RelGTTokens(_synthetic_graph(), task, table, k=6, num_workers=0)


def _collate_all(tokens: RelGTTokens) -> dict:
    return tokens.collate([tokens[i] for i in range(len(tokens))])


def test__tokenize__seed_token_is_index_zero(tokens: RelGTTokens) -> None:
    batch = _collate_all(tokens)
    b, k = batch["neighbor_types"].shape
    assert (b, k) == (8, 6)
    # Column 0 is always the seed: type t0, hop 0, relative time 0, the seed's own id.
    assert (batch["neighbor_types"][:, 0] == tokens.node_type_map["t0"]).all()
    assert (batch["neighbor_hops"][:, 0] == 0).all()
    assert (batch["neighbor_times"][:, 0] == 0).all()
    assert batch["neighbor_indices"][:, 0].tolist() == list(range(8))


def test__tokenize__sampled_neighbors_respect_seed_time(tokens: RelGTTokens) -> None:
    batch = _collate_all(tokens)
    # rel_time = (seed_time - neighbor_time) / day, so a real (hop 1/2) neighbor sampled
    # under the time filter has rel_time >= 0. (Fallback hop-3 tokens are exempt — they
    # are random padding for neighbor-poor seeds and may be negative.)
    real = (batch["neighbor_hops"] == 1) | (batch["neighbor_hops"] == 2)
    assert real.any()
    assert (batch["neighbor_times"][real] >= 0).all()


def test__tokenize__node_indices_are_global_ids(tokens: RelGTTokens) -> None:
    batch = _collate_all(tokens)
    # t0 is the first node type (offset 0), so a seed's global id is its local id.
    assert batch["node_indices"].tolist() == list(range(8))
    # Every (sample, token) slot belongs to exactly one node type's grouped frame.
    total_grouped = sum(len(v) for v in batch["grouped_indices"].values())
    assert total_grouped == batch["neighbor_types"].numel()


def test__tokenize__batch_feeds_relgt_forward(tokens: RelGTTokens) -> None:
    relgt = pytest.importorskip("relarena.models.relgt._vendor")
    from torch_frame.data.stats import StatType

    data = tokens.data
    col_names_dict = {nt: data[nt].tf.col_names_dict for nt in data.node_types}
    col_stats_dict = {
        nt: {c: {StatType.MEAN: 0.0, StatType.STD: 1.0} for c in ["c0", "c1"]}
        for nt in data.node_types
    }
    channels = 32
    model = relgt.RelGT(
        num_nodes=tokens.num_global_nodes,
        max_neighbor_hop=3,
        node_type_map=tokens.node_type_map,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_dict,
        local_num_layers=1,
        channels=channels,
        out_channels=1,
        global_dim=channels // 2,
        heads=2,
        conv_type="full",
        num_centroids=8,
        sample_node_len=tokens.k,
    )
    model.train()

    batch = _collate_all(tokens)
    out = model(
        neighbor_types=batch["neighbor_types"],
        node_indices=batch["node_indices"],
        neighbor_hops=batch["neighbor_hops"],
        neighbor_times=batch["neighbor_times"],
        grouped_tf_dict={
            "grouped_tfs": batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx": batch["flat_batch_idx"],
            "flat_nbr_idx": batch["flat_nbr_idx"],
        },
        edge_index=batch["edge_index"],
        batch=batch["batch"],
    )
    assert out.shape == (len(tokens), 1)
    assert torch.isfinite(out).all()


# -- HDF5 token cache --------------------------------------------------------

_TOKEN_KEYS = (
    "neighbor_types",
    "neighbor_indices",
    "neighbor_hops",
    "neighbor_times",
    "edge_index",
)


def _full_batch(t: RelGTTokens) -> dict:
    return t.collate([t[i] for i in range(len(t))])


def test__cache__roundtrip_reads_identical_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _synthetic_graph()
    task, table = _task_and_table(list(range(8)))

    first = RelGTTokens(
        graph,
        task,
        table,
        k=6,
        cache=CacheConfig(tmp_path, "fill"),
        run_identity=_identity(),
        num_workers=0,
    )
    written = list(tmp_path.rglob("*.h5"))
    assert len(written) == 1  # sampling persisted exactly one cache entry

    # A second instance must read the cache, NOT re-sample: break the sampler so any
    # attempt to sample would raise, then confirm construction still succeeds.
    import relarena.models.relgt.tokenize as tok

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("cache hit must not re-sample")

    monkeypatch.setattr(tok, "_sample_write_chunked", _boom)
    second = RelGTTokens(
        graph,
        task,
        table,
        k=6,
        cache=CacheConfig(tmp_path, "raise"),
        run_identity=_identity(),
        num_workers=0,
    )

    b1, b2 = _full_batch(first), _full_batch(second)
    for key in _TOKEN_KEYS:
        assert torch.equal(b1[key], b2[key])  # bit-for-bit identical tokens
    assert list(tmp_path.rglob("*.h5")) == written


def test__cache__different_graphs_get_distinct_entries(tmp_path: Path) -> None:
    task, table = _task_and_table(list(range(8)))
    # Same seeds, two differently-censored graphs (as inner vs outer would be): the
    # cache key fingerprints the graph, so they must not collide onto one file.
    for graph in (_synthetic_graph(n1=25), _synthetic_graph(n1=40)):
        RelGTTokens(
            graph,
            task,
            table,
            k=6,
            cache=CacheConfig(tmp_path, "fill"),
            run_identity=_identity(),
        )
    assert len(list(tmp_path.rglob("*.h5"))) == 2


def test__sampling__parallel_matches_serial() -> None:
    # Workers inherit the graph via fork (not per-task pickling) and each seed seeds
    # its own RNG, so parallel sampling must produce byte-identical tokens to serial.
    graph = _synthetic_graph()
    task, table = _task_and_table(list(range(8)))
    serial = RelGTTokens(graph, task, table, k=6, num_workers=0)
    parallel = RelGTTokens(graph, task, table, k=6, num_workers=2)

    bs = serial.collate([serial[i] for i in range(len(serial))])
    bp = parallel.collate([parallel[i] for i in range(len(parallel))])
    for key in _TOKEN_KEYS:
        assert torch.equal(bs[key], bp[key])


def test__sampling_seed__different_hash_secrets__is_process_stable() -> None:
    code = (
        "from relarena.models.relgt.tokenize import _sampling_seed; "
        "print(_sampling_seed('t0', 7, 1577836800.0, 6))"
    )
    outputs = []
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test__token_cache_key__tracks_task_seed_sampler_and_phase_inputs() -> None:
    graph = _synthetic_graph()
    task, table = _task_and_table(list(range(8)))
    table_input = get_node_train_table_input(table=table, task=task)
    _, node_idxs = table_input.nodes
    identity = _identity()
    key = token_cache_key(graph, task, node_idxs, table_input.time, 6, True, identity)
    assert (
        token_cache_key(graph, task, node_idxs, table_input.time, 7, True, identity)
        != key
    )
    assert (
        token_cache_key(graph, task, node_idxs, table_input.time, 6, False, identity)
        != key
    )
    assert (
        token_cache_key(
            graph,
            task,
            node_idxs,
            table_input.time,
            6,
            True,
            identity.for_phase("outer"),
        )
        != key
    )
    assert str(key).endswith("k-6-undirected-1.h5")


def test__persistent_tokens__missing_run_identity__raises(tmp_path: Path) -> None:
    task, table = _task_and_table(list(range(2)))
    with pytest.raises(ValueError, match="RunIdentity"):
        RelGTTokens(
            _synthetic_graph(),
            task,
            table,
            k=6,
            cache=CacheConfig(tmp_path, "fill"),
        )


def test__cache__corrupt_shape__fails_validation(tmp_path: Path) -> None:
    import h5py

    graph = _synthetic_graph()
    task, table = _task_and_table(list(range(2)))
    table_input = get_node_train_table_input(table=table, task=task)
    _, node_idxs = table_input.nodes
    key = token_cache_key(
        graph, task, node_idxs, table_input.time, 6, True, _identity()
    )
    path = tmp_path / key
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        for name in ("types", "indices", "hops", "times"):
            handle.create_dataset(name, data=[[0]])
        handle.create_dataset("edges", shape=(2, 0), dtype="int32")
        handle.create_dataset("edges_offsets", data=[0, 0])

    with pytest.raises(ValueError, match="corrupt RelGT tokens"):
        RelGTTokens(
            graph,
            task,
            table,
            k=6,
            cache=CacheConfig(tmp_path, "raise"),
            run_identity=_identity(),
        )


def test__precompute_tokens__writes_one_entry_per_table(tmp_path: Path) -> None:
    graph = _synthetic_graph()
    task, table_a = _task_and_table(list(range(8)))
    _, table_b = _task_and_table(list(range(8, 16)))  # different seeds
    precompute_tokens(
        graph,
        task,
        [table_a, table_b],
        k=6,
        cache=CacheConfig(tmp_path, "fill"),
        run_identity=_identity(),
    )
    assert len(list(tmp_path.rglob("*.h5"))) == 2
