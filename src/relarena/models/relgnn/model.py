"""RelGNN — composite message passing over atomic routes (Chen et al. 2025).

RelGNN (https://arxiv.org/abs/2502.06784) is a drop-in replacement for the GNN inside
the *same* RDL pipeline as the `graphsage` baseline: the relational DB becomes a
heterogeneous temporal graph, but instead of plain neighbour aggregation it message-
passes over **atomic routes** — direct source->destination paths derived from the
foreign-key schema (`dim-dim` single fkeys and `dim-fact-dim` two-fkey compositions
through a shared fact table) — with a per-route `TransformerConv`. The vendored model
+ route builder live under `_relgnn`; this module is the relarena adapter
(fit/predict, search space, graph cache), mirroring `graphsage.py`.

**Hyperparameters.** RelGNN's paper hand-tunes a *separate* config per (dataset, task);
there is no single reported default. We instead expose a unified search space and a
default config taken from the *modal* per-task values across the 21 RelBench entity
tasks: `channels=128`, `num_model_layers=2`, `num_heads=8`, `aggr=sum`,
`num_neighbors=128`, `subgraph_type=directional` (`simplified_MP` — a one-off
conv variant used on a single task — stays `False` and is not searched).
`num_model_layers` (GNN depth) is decoupled from the neighbour-sampling depth, which
is fixed at 2 as in the paper. The training schedule (batch size, epochs, steps) is
fixed infra, not tuned.

**Graph cache.** `preprocessing.py` owns its complete versioned directory key,
RelBench codec, integrity check, and completion marker. The key depends on the
censored database and phase but deliberately omits task and model settings, so every
task over one dataset shares the same inner/outer graphs. The caller's explicit
cache policy determines whether a miss raises, fills, or computes privately. Warm it
with `python -m relarena.models.relgnn.warm_cache`.

GPU recommended. Heavy deps (PyG, PyTorch Frame, the vendored model) are imported lazily
inside `fit` / `predict`, so importing this module to register the model is cheap.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np
import torch
from ConfigSpace import Categorical, ConfigurationSpace, Float
from relbench.base import Database, EntityTask, Table

from relarena.metrics import get_metric, primary_metric
from relarena.model import RelArenaModel
from relarena.models._shared.gnn.training import (
    default_device,
    infer,
    task_setup,
    train_epoch,
)
from relarena.models.relgnn.preprocessing import load_graph
from relarena.registry import register_model
from relarena.search_space import SearchSpace
from relarena.tasks import ENTITY_TASK_TYPES

# Fixed budget / infra constants (never tuned; not part of any hyperparameter config).
_BATCH_SIZE = 512
_EPOCHS = 10
_MAX_STEPS_PER_EPOCH = 2000
# Neighbour-sampling depth: fixed at 2 across the paper's entity tasks, and decoupled
# from the (tuned) GNN depth `num_model_layers`.
_SAMPLING_DEPTH = 2
_TEMPORAL_STRATEGY = "uniform"


def _relgnn_config_space() -> ConfigurationSpace:
    """The RelGNN search space: capacity / receptive field / attention / optimization.

    Discrete sets for the structural knobs (values drawn from those the paper used
    across tasks); `lr` is log-uniform. `simplified_MP` is intentionally excluded —
    it is a per-task conv variant used on a single task and stays at its `False`
    default.
    """
    return ConfigurationSpace(
        space=[
            Categorical("channels", [64, 128, 256]),
            Categorical("num_model_layers", [1, 2, 4]),
            Categorical("num_heads", [2, 4, 8, 16]),
            Categorical("aggr", ["sum", "mean"]),
            Categorical("num_neighbors", [64, 128, 256]),
            Categorical("subgraph_type", ["directional", "bidirectional"]),
            Float("lr", (1e-3, 1e-2), log=True),
        ],
    )


#: The default ("zero-tuning") regime: the modal per-task config across the 21 RelBench
#: entity tasks (see module docstring). Passed explicitly so the default trial records
#: the real config, and so `simplified_MP` (default-only, not searched) is present.
_DEFAULT_CONFIG = {
    "channels": 128,
    "num_model_layers": 2,
    "num_heads": 8,
    "aggr": "sum",
    "num_neighbors": 128,
    "subgraph_type": "directional",
    "simplified_MP": False,
    "lr": 0.005,
}

#: RelGNN's search space: the space above + the explicit default config. `configs`
#: runs the default config first, then `n_trials` samples.
RELGNN_SPACE = SearchSpace(
    space=_relgnn_config_space(), default_overrides=_DEFAULT_CONFIG
)


def _make_loader(
    data: Any,
    table: Table,
    task: EntityTask,
    *,
    num_neighbors: int,
    subgraph_type: str,
    batch_size: int,
    shuffle: bool,
) -> Any:
    """Build a time-aware `NeighborLoader` over `table`'s seed entities.

    Per-layer fan-out decays geometrically over the fixed sampling depth
    (`num_neighbors / 2**i`). `subgraph_type` (`directional` / `bidirectional`)
    is a RelGNN loader knob: it controls whether reverse edges are added to the sampled
    subgraph, which atomic routes can rely on. `shuffle=False` preserves row order so
    concatenated predictions align with the rows `EntityTask.evaluate` expects.
    """
    from relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader

    table_input = get_node_train_table_input(table=table, task=task)
    return NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(_SAMPLING_DEPTH)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        subgraph_type=subgraph_type,
        batch_size=batch_size,
        temporal_strategy=_TEMPORAL_STRATEGY,
        shuffle=shuffle,
        num_workers=0,
    )


@register_model(search_space=RELGNN_SPACE)
class RelGNNModel(RelArenaModel):
    """RelGNN composite message passing on the RelBench relational graph.

    Tuned over `RELGNN_SPACE` (channels, GNN depth, attention heads, aggr, neighbour
    fan-out, subgraph type, lr); the default regime runs the modal per-task config.
    """

    name = "relgnn"

    # Multiclass omitted on purpose — no RelBench v1 entity task is multiclass.
    supported_task_types = ENTITY_TASK_TYPES

    # -- fit / predict --------------------------------------------------------

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Load the graph, then train RelGNN (best-checkpoint on val if given).

        `val_table` is `None` on the refit split — then we train the full fixed
        epoch budget. `time_limit` (seconds, soft) ends the epoch loop early if
        exceeded.
        """
        from torch_geometric.seed import seed_everything

        from relarena.models.relgnn._vendor.atomic_routes import get_atomic_routes
        from relarena.models.relgnn._vendor.model import RelGNN_Model

        seed_everything(seed)
        device = default_device()
        data, col_stats_dict = load_graph(db, self.cache, self.run_identity)
        atomic_routes = get_atomic_routes(data.edge_types)

        channels = int(self.config["channels"])
        num_model_layers = int(self.config["num_model_layers"])
        num_heads = int(self.config["num_heads"])
        aggr = str(self.config["aggr"])
        num_neighbors = int(self.config["num_neighbors"])
        subgraph_type = str(self.config["subgraph_type"])
        # `simplified_MP` is default-only (not in the search space), so sampled
        # configs omit it — fall back to the default-regime value.
        simplified_MP = bool(self.config.get("simplified_MP", False))
        lr = float(self.config["lr"])

        out_channels, loss_fn, clamp = task_setup(task, train_table)

        model = RelGNN_Model(
            data,
            col_stats_dict,
            num_model_layers=num_model_layers,
            channels=channels,
            out_channels=out_channels,
            aggr=aggr,
            norm="batch_norm",
            atomic_routes=atomic_routes,
            num_heads=num_heads,
            simplified_MP=simplified_MP,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = _make_loader(
            data,
            train_table,
            task,
            num_neighbors=num_neighbors,
            subgraph_type=subgraph_type,
            batch_size=_BATCH_SIZE,
            shuffle=True,
        )

        # Early stopping on val (tuning split): score on the harness selection metric
        # so the kept checkpoint matches how the runner picks configs.
        val_loader = None
        metric_fn = primary_metric(task)
        higher = get_metric(metric_fn).higher_is_better
        best_score = -math.inf if higher else math.inf
        best_state: dict[str, Any] | None = None
        if val_table is not None:
            val_loader = _make_loader(
                data,
                val_table,
                task,
                num_neighbors=num_neighbors,
                subgraph_type=subgraph_type,
                batch_size=_BATCH_SIZE,
                shuffle=False,
            )
            val_true = val_table.df[task.target_col].to_numpy()

        start = time.monotonic()
        for _ in range(_EPOCHS):
            train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                task,
                device,
                _MAX_STEPS_PER_EPOCH,
            )
            if val_loader is not None:
                val_pred = infer(model, val_loader, task, device, clamp)
                score = float(metric_fn(val_true, val_pred))
                if np.isfinite(score) and (
                    (higher and score >= best_score)
                    or (not higher and score <= best_score)
                ):
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
            if time_limit is not None and time.monotonic() - start > time_limit:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self._device = device
        self._clamp = clamp
        self._num_neighbors = num_neighbors
        self._subgraph_type = subgraph_type

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Run RelGNN over `table`; return the evaluate-contract array."""
        data, _ = load_graph(db, self.cache, self.run_identity)
        loader = _make_loader(
            data,
            table,
            task,
            num_neighbors=self._num_neighbors,
            subgraph_type=self._subgraph_type,
            batch_size=_BATCH_SIZE,
            shuffle=False,
        )
        return infer(self._model, loader, task, self._device, self._clamp)


@register_model(search_space=RELGNN_SPACE)
class RelGNNEarlyStopModel(RelGNNModel):
    """RelGNN reported at its best-val checkpoint instead of a train+val refit.

    Identical to `RelGNNModel` except the final test model is the early-stopped
    best-val checkpoint (`refit_on_full_data = False`): train on `train` alone,
    keep the checkpoint that scores best on `val`, and predict test — the protocol
    many GNN papers report. `fit` already keeps the best-val checkpoint whenever a
    `val_table` is passed, so no training change is needed.
    """

    name = "relgnn-es"
    refit_on_full_data = False
