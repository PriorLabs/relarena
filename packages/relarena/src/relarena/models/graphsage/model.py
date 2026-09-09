"""Heterogeneous GraphSAGE baseline — RelBench's relational-deep-learning GNN.

GraphSAGE (Hamilton et al. 2017, https://arxiv.org/abs/1706.02216) on the
*heterogeneous, temporal* graph RelBench builds from a relational database: every
table is a node type, foreign keys are edges, and each entity's prediction is read
out from a message-passing GNN seeded at that entity's row and time. Architecture
(all from RelBench's RDL stack):

  * **encode** — per-node-type PyTorch Frame column encoders (`HeteroEncoder`)
    turn each table's raw columns into `channels`-dim node embeddings; a
    `HeteroTemporalEncoder` adds a relative-time signal (seed minus neighbor time);
  * **message-pass** — `HeteroGraphSAGE`: `num_layers` rounds of per-edge-type
    `SAGEConv` (neighbor `aggr`) summed across edge types, with LayerNorm + ReLU;
  * **read out** — a 1-layer MLP head over the seed nodes, sized per task type.

Why it's in RelArena: RelBench / KumoRFMv2 report GraphSAGE only at its **default
config** (no HPO). RelArena's job is fair, budget-matched evaluation, so we (a)
reproduce that default baseline — the default regime runs RelBench's exact un-tuned
config (`_DEFAULT_CONFIG`) — and (b) expose a **small search space**
so we can ask whether GraphSAGE benefits from tuning. The tuned knobs are the ones
that change GraphSAGE's capacity / receptive field / optimization without changing
the architecture: hidden `channels`, `num_layers`, neighbor `aggr` (the paper's
aggregator choice), the per-layer neighbor sample size `num_neighbors` (the paper's
`S`), and `lr`. (No dropout knob: `HeteroGraphSAGE` has none — dropout is a
GAT-only parameter upstream.)

Temporal correctness is double-covered: the harness hands `fit` / `predict` a
database already censored at the split's cutoff, *and* the loader samples only
neighbors before each seed node's timestamp (`time_attr="time"`). On the tuning
(inner) split `val_table` is used for best-checkpoint early stopping; on the refit
(outer) split it is `None` and we fall back to a fixed epoch budget.

GPU recommended (training is GNN message passing). Heavy deps (PyG, PyTorch Frame,
sentence-transformers) live in the `graphsage` extra and are imported lazily, so
importing this module to register the model is cheap.

The GNN `Model` + `GloveTextEmbedding` are vendored verbatim in
`models/_shared/gnn/_vendor/gnn.py` (from snap-stanford/relbench `examples/` @
74d4c37). This module is the relarena
adapter: it imports them, wires RelBench's `relbench.modeling` graph / loader helpers,
and adds the fit/predict contract, the search space, and a per-db graph cache. The
training loop here is adapted from `examples/gnn_entity.py`.
"""

from __future__ import annotations

import copy
import gc
import logging
import math
import os
import time
from typing import Any, Callable

import numpy as np
import torch
from ConfigSpace import Categorical, ConfigurationSpace, Float
from relbench.base import Database, EntityTask, Table, TaskType

from relarena.metrics import get_metric, primary_metric
from relarena.model import RelArenaModel
from relarena.models._shared.gnn.graph import GRAPH_CACHE, build_graph
from relarena.models._shared.gnn.training import (
    default_device,
    infer,
    task_setup,
    train_epoch,
)
from relarena.registry import register_model
from relarena.search_space import SearchSpace

logger = logging.getLogger(__name__)

# Cut CUDA fragmentation OOMs on the bigger configs. The caching allocator reads this
# on its first allocation (only later, inside fit/predict), so setting it here is early
# enough; setdefault lets an explicit environment override win.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Fixed budget / infra constants (never tuned; not part of any hyperparameter config).
_BATCH_SIZE = 512
# OOM floor: on a CUDA OutOfMemoryError we halve the batch size and retry; below this we
# give up and re-raise, so a genuinely too-large config fails loudly instead of looping.
_MIN_BATCH_SIZE = 16
_EPOCHS = 10
_MAX_STEPS_PER_EPOCH = 2000
_TEMPORAL_STRATEGY = "uniform"
# Regression predictions are clamped to this train-target percentile range at inference
# (matches RelBench, which clips to [2, 98] to keep MAE robust to tail extrapolation).


def _graphsage_config_space() -> ConfigurationSpace:
    """The small GraphSAGE search space: capacity / receptive field / optimization.

    Discrete sets (not ranges) for the structural knobs so trials land on sensible
    powers/values; `lr` is log-uniform. `default_overrides={}` (not a config drawn
    here) defines the reported default regime, so these are purely the tuning samples.
    """
    return ConfigurationSpace(
        space=[
            Categorical("channels", [64, 128, 256]),
            Categorical("num_layers", [2, 3]),
            Categorical("aggr", ["mean", "sum", "max"]),
            Categorical("num_neighbors", [64, 128, 256]),
            Float("lr", (1e-3, 1e-2), log=True),
        ],
    )


#: RelBench's un-tuned GraphSAGE defaults (gnn_entity.py argparse) — the default
#: regime, passed explicitly so the default trial's *recorded* config is the real config
#: (not `{}`) and the values live in one obvious place (cf. rdblearn).
_DEFAULT_CONFIG = {
    "channels": 128,
    "num_layers": 2,
    "aggr": "sum",
    "num_neighbors": 128,
    "lr": 0.005,
}

#: GraphSAGE's search space: the small space above + the explicit default config.
#: `SearchSpace.configs` runs the default config first, then `n_trials` samples.
GRAPHSAGE_SPACE = SearchSpace(
    space=_graphsage_config_space(), default_overrides=_DEFAULT_CONFIG
)


def _cuda_cleanup(device: str) -> None:
    """Release CUDA memory left dangling after an OOM so a retry can allocate."""
    # gc first so tensors held by the exception traceback are dropped, then empty_cache
    # returns their blocks; synchronize so in-flight failed kernels are fully torn down.
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_with_oom_retry(
    attempt: Callable[[int], Any],
    *,
    start_batch_size: int,
    device: str,
    what: str,
) -> tuple[Any, int]:
    """Call `attempt(batch_size)`, halving the batch size on CUDA OOM.

    Returns `(result, batch_size)` for the batch size that succeeded. Halves down to
    `_MIN_BATCH_SIZE` (RelBench adapts batch size the same way) and re-raises if OOM
    persists at the floor, so a too-large config surfaces a real failure. Only
    `torch.OutOfMemoryError` is caught — other errors propagate untouched.

    Args:
        attempt: Trains/infers from scratch at the given batch size; raises
            `torch.OutOfMemoryError` if it overflows the GPU.
        start_batch_size: Batch size for the first attempt.
        device: Torch device string, for the inter-attempt CUDA cleanup.
        what: Label for the downshift log lines (e.g. `"fit"` / `"predict"`).
    """
    batch_size = start_batch_size
    while True:
        try:
            return attempt(batch_size), batch_size
        except torch.OutOfMemoryError:
            _cuda_cleanup(device)
            if batch_size <= _MIN_BATCH_SIZE:
                logger.warning(
                    "graphsage %s OOM at floor batch_size=%d; re-raising",
                    what,
                    batch_size,
                )
                raise
            new_batch_size = max(batch_size // 2, _MIN_BATCH_SIZE)
            logger.warning(
                "graphsage %s OOM at batch_size=%d; retrying at batch_size=%d",
                what,
                batch_size,
                new_batch_size,
            )
            batch_size = new_batch_size


#: Module-level singleton — one censored DB's graph at a time (see `DBGraphCache`).
def _make_loader(
    data: Any,
    table: Table,
    task: EntityTask,
    *,
    num_neighbors: int,
    num_layers: int,
    batch_size: int,
    shuffle: bool,
) -> Any:
    """Build a time-aware `NeighborLoader` over `table`'s seed entities.

    Per-layer fan-out decays geometrically (`num_neighbors / 2**layer`, RelBench's
    convention). `shuffle=False` preserves `table` row order, so concatenated
    predictions align with the rows `EntityTask.evaluate` expects.
    """
    from relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader

    table_input = get_node_train_table_input(table=table, task=task)
    return NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=batch_size,
        temporal_strategy=_TEMPORAL_STRATEGY,
        shuffle=shuffle,
        num_workers=0,
    )


@register_model(search_space=GRAPHSAGE_SPACE)
class GraphSAGEModel(RelArenaModel):
    """Heterogeneous GraphSAGE on the RelBench relational graph.

    Tuned over `GRAPHSAGE_SPACE` (channels, layers, aggr, neighbor fan-out, lr);
    the default regime reproduces RelBench's un-tuned baseline.
    """

    name = "graphsage"

    # Multiclass omitted on purpose — no RelBench v1 entity task is multiclass, so the
    # path is untested; the runner will skip any multiclass task for this model.
    supported_task_types = frozenset(
        {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.REGRESSION,
        }
    )

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
        """Build the graph, then train GraphSAGE (best-checkpoint on val if given).

        `val_table` is `None` on the refit split — then we train the full fixed
        epoch budget. `time_limit` (seconds, soft) ends the epoch loop early if
        exceeded. On a CUDA OOM, training restarts from scratch at half the batch size
        (down to `_MIN_BATCH_SIZE`) so big configs complete instead of being dropped.
        """
        from torch_geometric.seed import seed_everything

        device = default_device()
        data, col_stats_dict = GRAPH_CACHE.get(db, lambda: build_graph(db, device))

        # Every config (the explicit default + the sampled ones) carries all five
        # tuned keys, so we read them directly rather than via fallbacks.
        channels = int(self.config["channels"])
        num_layers = int(self.config["num_layers"])
        aggr = str(self.config["aggr"])
        lr = float(self.config["lr"])
        num_neighbors = int(self.config["num_neighbors"])

        out_channels, loss_fn, clamp = task_setup(task, train_table)

        # The GNN model is vendored verbatim in models/_shared/gnn/_vendor/gnn.py
        # (RelBench's examples/model.py); we drive its SAGE path with our resolved
        # config.
        from relarena.models._shared.gnn._vendor.gnn import Model

        def attempt(batch_size: int) -> Any:
            # Rebuild model/optimizer/loaders from scratch each try: a mid-epoch OOM
            # leaves them in an indeterminate state. Re-seed first so a given batch size
            # is deterministic regardless of how many OOM retries preceded it.
            seed_everything(seed)
            model = Model(
                data,
                col_stats_dict,
                num_layers=num_layers,
                channels=channels,
                out_channels=out_channels,
                aggr=aggr,
                norm="batch_norm",
                gnn="sage",
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_loader = _make_loader(
                data,
                train_table,
                task,
                num_neighbors=num_neighbors,
                num_layers=num_layers,
                batch_size=batch_size,
                shuffle=True,
            )

            # Early stopping on val (tuning split): score on the harness selection
            # metric so the kept checkpoint matches how the runner picks configs.
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
                    num_layers=num_layers,
                    batch_size=batch_size,
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
            return model

        model, batch_size = _run_with_oom_retry(
            attempt, start_batch_size=_BATCH_SIZE, device=device, what="fit"
        )

        self._model = model
        self._device = device
        self._clamp = clamp
        self._num_neighbors = num_neighbors
        self._num_layers = num_layers
        self._batch_size = batch_size

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Run GraphSAGE over `table`; return the evaluate-contract array."""
        data, _ = GRAPH_CACHE.get(db, lambda: build_graph(db, self._device))

        def attempt(batch_size: int) -> np.ndarray:
            loader = _make_loader(
                data,
                table,
                task,
                num_neighbors=self._num_neighbors,
                num_layers=self._num_layers,
                batch_size=batch_size,
                shuffle=False,
            )
            return infer(self._model, loader, task, self._device, self._clamp)

        # Start from the batch size fit settled on (a good upper bound for inference).
        # TODO: this retry is probably unnecessary. Inference keeps no gradients or
        # optimizer state, so it should not OOM at a batch size training survived.
        # Consider calling attempt(self._batch_size) directly and dropping the wrapper.
        preds, _ = _run_with_oom_retry(
            attempt,
            start_batch_size=self._batch_size,
            device=self._device,
            what="predict",
        )
        return preds
