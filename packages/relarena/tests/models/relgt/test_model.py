"""Tests for the RelGT relarena adapter (`models/relgt.py`).

Dep-free tests cover registration, the size-aware search-space factory, and the
training schedule. A fit→predict integration smoke (guarded by the `relgt` extra)
runs the real adapter end-to-end on a synthetic graph — `_build_graph` is patched so
no RelBench download / GloVe / GPU is needed — exercising the train loop,
best-checkpoint-on-val, and the table-row-aligned predict.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.relgt.model import (
    _DEFAULT_CONFIG,
    _LARGE_NODE_THRESHOLD,
    _MEDIUM_NODE_THRESHOLD,
    _full_grid,
    _medium_grid,
    _schedule,
    relgt_search_space,
)
from relarena.registry import registry
from relarena.search_space import TaskStats

_GRID_COMBOS = {(ly, dr) for ly in (1, 4, 8) for dr in (0.3, 0.4, 0.5)}


# -- registration & search space (dep-free) ---------------------------------


def test__registry__relgt_registered_with_factory() -> None:
    import relarena.models  # noqa: F401  (triggers registration)
    from relarena.models.relgt import RelGTModel

    assert "relgt" in registry
    assert registry.get("relgt") is RelGTModel
    # The registered "space" is the size-aware factory itself, not a fixed space.
    assert registry.search_space("relgt") is relgt_search_space


def test__relgt_model__supports_binary_and_regression_only() -> None:
    from relarena.models.relgt import RelGTModel

    assert RelGTModel.supported_task_types == frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )


def test__relgt_model__uses_best_val_checkpoint_not_full_refit() -> None:
    from relarena.models.relgt import RelGTModel

    # RelGT reports a best-val checkpoint (train-only), not a train+val refit.
    assert RelGTModel.refit_on_full_data is False


def test__relgt_search_space__small_task__full_grid_default_first() -> None:
    space = relgt_search_space(TaskStats(num_train_nodes=_MEDIUM_NODE_THRESHOLD))
    configs = space.configs(n_trials=99, seed=0)
    assert len(configs) == 9
    assert configs[0] == _DEFAULT_CONFIG  # default {L4, d0.3} runs first
    assert {(c["num_layers"], c["dropout"]) for c in configs} == _GRID_COMBOS


def test__relgt_search_space__medium_task__dropout_sweep_at_default_depth() -> None:
    space = relgt_search_space(TaskStats(num_train_nodes=_MEDIUM_NODE_THRESHOLD + 1))
    configs = space.configs(n_trials=99, seed=0)
    assert len(configs) == 3
    assert configs[0] == _DEFAULT_CONFIG  # default {L4, d0.3} runs first
    assert all(c["num_layers"] == _DEFAULT_CONFIG["num_layers"] for c in configs)
    assert {c["dropout"] for c in configs} == {0.3, 0.4, 0.5}
    assert all(c in _full_grid() for c in configs)  # nested subset


def test__relgt_search_space__large_task__single_l4_config() -> None:
    space = relgt_search_space(TaskStats(num_train_nodes=_LARGE_NODE_THRESHOLD + 1))
    configs = space.configs(n_trials=99, seed=0)
    assert configs == [_DEFAULT_CONFIG]  # >1M nodes: fix L=4, one config


def test__full_grid__nine_configs_default_first_and_in_grid() -> None:
    grid = _full_grid()
    assert len(grid) == 9
    assert grid[0] == _DEFAULT_CONFIG
    assert _DEFAULT_CONFIG in grid


def test__medium_grid__three_dropout_configs_at_default_depth() -> None:
    grid = _medium_grid()
    assert len(grid) == 3
    assert grid[0] == _DEFAULT_CONFIG
    assert all(c["num_layers"] == _DEFAULT_CONFIG["num_layers"] for c in grid)


def test__schedule__threshold_picks_small_or_large() -> None:
    # At/below the threshold → small-data schedule; strictly above → large.
    assert _schedule(_LARGE_NODE_THRESHOLD) == (256, 100, 3000)
    assert _schedule(_LARGE_NODE_THRESHOLD + 1) == (1024, 10, 500)


# -- fit/predict integration smoke (needs the relgt extra) ------------------

_BASE = 1_577_836_800  # 2020-01-01 UTC, UNIX seconds
_DAY = 86_400


def _synthetic_graph_and_stats() -> tuple:
    """Tiny heterogeneous temporal graph + per-type col stats (no download)."""
    pytest.importorskip("torch_geometric")
    import torch
    from torch_frame import TensorFrame, stype
    from torch_frame.data.stats import StatType
    from torch_geometric.data import HeteroData

    n0, n1 = 16, 24
    data = HeteroData()
    data["t0"].num_nodes = n0
    data["t1"].num_nodes = n1
    data["t0"].time = torch.tensor([_BASE + i * _DAY for i in range(n0)])
    data["t1"].time = torch.tensor([_BASE + j * _DAY for j in range(n1)])
    for nt, n in (("t0", n0), ("t1", n1)):
        data[nt].tf = TensorFrame(
            feat_dict={stype.numerical: torch.randn(n, 2)},
            col_names_dict={stype.numerical: ["a", "b"]},
        )
    src = [i for i in range(n0) for _ in range(3)]
    dst = [(i + d) % n1 for i in range(n0) for d in range(3)]
    data["t0", "to", "t1"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    col_stats = {
        nt: {c: {StatType.MEAN: 0.0, StatType.STD: 1.0} for c in ("a", "b")}
        for nt in ("t0", "t1")
    }
    return data, col_stats


def _task_and_tables() -> tuple:
    task = SimpleNamespace(
        entity_table="t0",
        entity_col="id",
        target_col="y",
        time_col="ts",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    def table(ids: list[int]) -> Any:
        from relbench.base import Table

        cutoffs = [pd.Timestamp(_BASE + (i + 5) * _DAY, unit="s") for i in ids]
        df = pd.DataFrame({"id": ids, "ts": cutoffs, "y": [i % 2 for i in ids]})
        return Table(df=df, fkey_col_to_pkey_table={}, pkey_col=None, time_col="ts")

    return task, table([0, 1, 2, 3, 4, 5, 6, 7]), table([8, 9, 10, 11, 12, 13])


def test__relgt_model__fit_predict__synthetic_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch_geometric")
    from relarena.models._shared.gnn import graph as shared_graph
    from relarena.models._shared.gnn.graph_cache import DBGraphCache
    from relarena.models.relgt import model as relgt

    data, col_stats = _synthetic_graph_and_stats()
    task, train_table, val_table = _task_and_tables()

    # Patch graph materialization (no download/GloVe) and shrink the architecture +
    # schedule so the real fit/predict path runs in a couple of seconds on CPU.
    monkeypatch.setattr(
        shared_graph, "build_graph", lambda db, device: (data, col_stats)
    )
    monkeypatch.setattr(shared_graph, "GRAPH_CACHE", DBGraphCache())
    monkeypatch.setattr(relgt, "_CHANNELS", 16)
    monkeypatch.setattr(relgt, "_NUM_HEADS", 2)
    monkeypatch.setattr(relgt, "_NUM_NEIGHBORS", 6)
    monkeypatch.setattr(relgt, "_NUM_CENTROIDS", 8)
    monkeypatch.setattr(relgt, "_SMALL_SCHEDULE", (4, 1, 2))
    monkeypatch.setenv("RELARENA_CACHE_DIR", str(tmp_path))

    model = relgt.RelGTModel(dict(_DEFAULT_CONFIG))
    model.fit(task, object(), train_table, val_table, seed=0)
    pred = model.predict(task, object(), val_table)

    assert pred.shape == (len(val_table.df),)
    assert np.isfinite(pred).all()
    # Binary task → predictions are probabilities.
    assert pred.min() >= 0.0 and pred.max() <= 1.0
