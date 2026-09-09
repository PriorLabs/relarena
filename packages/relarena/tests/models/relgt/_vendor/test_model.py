"""Smoke test for the vendored RelGT model (`models/relgt/_vendor`).

Exercises the real `RelGT.forward` — all five token encoders (type, hop, time,
features, PE), the local+global attention layer, and the prediction head — on a
synthetic single-node-type batch, so a regression in the vendored copy (or a broken
inter-module import) fails here without needing a RelBench download. The data-driven
path (real tokenization / graph) is covered by the tokenizer tests and the CLI smoke.
"""

from __future__ import annotations

import pytest

# Accessed via the importorskip handle (not `from ... import`) so the module-level
# imports stay at the top and the test skips cleanly without the `relgt` extra.
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
torch_frame = pytest.importorskip("torch_frame")
StatType = pytest.importorskip("torch_frame.data.stats").StatType
RelGT = pytest.importorskip("relarena.models.relgt._vendor").RelGT

TensorFrame = torch_frame.TensorFrame
stype = torch_frame.stype


def _synthetic_batch(
    *, batch_size: int, k: int, n_numerical: int
) -> tuple[dict, dict, dict, dict]:
    """Build inputs for `RelGT.forward` with a single node type `t0`.

    All `batch_size * k` tokens are type 0 with `n_numerical` numerical columns;
    the seed token sits at column 0 (hop 0, time 0). Returns the kwargs dict plus the
    `node_type_map` / column metadata the model is constructed from.
    """
    node_type_map = {"t0": 0}
    col_names_dict = {"t0": {stype.numerical: [f"c{i}" for i in range(n_numerical)]}}
    col_stats_dict = {
        "t0": {
            f"c{i}": {StatType.MEAN: 0.0, StatType.STD: 1.0} for i in range(n_numerical)
        }
    }

    total = batch_size * k
    neighbor_types = torch.zeros(batch_size, k, dtype=torch.long)
    node_indices = torch.arange(batch_size, dtype=torch.long)
    neighbor_hops = torch.zeros(batch_size, k, dtype=torch.long)
    neighbor_hops[:, 1:] = torch.randint(1, 3, (batch_size, k - 1))
    neighbor_times = torch.zeros(batch_size, k, dtype=torch.float)
    neighbor_times[:, 1:] = torch.rand(batch_size, k - 1) * 30.0

    # One concatenated TorchFrame holding every token (all type 0), with the flat
    # scatter indices the encoder uses to rebuild the [B, K, channels] tensor.
    big_tf = TensorFrame(
        feat_dict={stype.numerical: torch.randn(total, n_numerical)},
        col_names_dict={stype.numerical: [f"c{i}" for i in range(n_numerical)]},
    )
    grouped_tf_dict = {
        "grouped_tfs": {0: big_tf},
        "grouped_indices": {0: list(range(total))},
        "flat_batch_idx": [b for b in range(batch_size) for _ in range(k)],
        "flat_nbr_idx": [j for _ in range(batch_size) for j in range(k)],
    }

    # Per-sample star subgraph (seed -> neighbors), batched with a node offset; the PE
    # encoder reshapes [total_nodes, .] back to [B, K, .] so batch must be K-uniform.
    edges = []
    for b in range(batch_size):
        off = b * k
        for j in range(1, k):
            edges += [(off, off + j), (off + j, off)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.arange(batch_size).repeat_interleave(k)

    kwargs = dict(
        neighbor_types=neighbor_types,
        node_indices=node_indices,
        neighbor_hops=neighbor_hops,
        neighbor_times=neighbor_times,
        grouped_tf_dict=grouped_tf_dict,
        edge_index=edge_index,
        batch=batch,
    )
    return kwargs, node_type_map, col_names_dict, col_stats_dict


@pytest.mark.parametrize("out_channels", [1, 3])
def test__relgt_forward__synthetic_single_type_batch__returns_logits(
    out_channels: int,
) -> None:
    batch_size, k, channels = 2, 4, 32
    kwargs, node_type_map, col_names_dict, col_stats_dict = _synthetic_batch(
        batch_size=batch_size, k=k, n_numerical=2
    )

    model = RelGT(
        num_nodes=64,
        max_neighbor_hop=3,
        node_type_map=node_type_map,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_dict,
        local_num_layers=1,
        channels=channels,
        out_channels=out_channels,
        global_dim=channels // 2,
        heads=2,
        conv_type="full",
        num_centroids=8,
        sample_node_len=k,
    )
    model.train()  # global path updates the EMA codebook in train mode

    out = model(**kwargs)

    assert out.shape == (batch_size, out_channels)
    assert torch.isfinite(out).all()
