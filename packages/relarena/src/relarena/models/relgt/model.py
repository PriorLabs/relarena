"""Relational Graph Transformer (RelGT) as a relarena method.

RelGT (Dwivedi et al., https://arxiv.org/abs/2505.10960) drops in for the GNN within
the *same* RDL pipeline as the `graphsage` baseline: the relational DB becomes a
heterogeneous temporal graph (`make_pkey_fkey_graph`, reused from `graphsage`),
each seed entity is turned into a fixed-length token sequence (seed + sampled
neighbours, five components each — features/type/hop/time/local-PE), and a transformer
with local attention over the tokens plus global attention to a learnable codebook
reads out the prediction. The model + sampler are vendored verbatim under `_relgt`;
this module is the relarena adapter (fit/predict + search space), mirroring
`graphsage.py` — including reusing its graph cache, graph builder, device pick, and
per-task-type setup.

**Size-aware search space (the paper's regime).** The RelGT paper tunes
`L ∈ {1,4,8} × dropout ∈ {0.3,0.4,0.5}` on datasets under ~1M training nodes, but
fixes `L=4` on the larger ones "due to compute budgets". That is genuinely a
*search space that depends on dataset scale*, so we register a factory
(`relgt_search_space`) rather than smuggle the choice into the trial budget: it
returns the full 9-config grid for small tasks and a single `{L=4, dropout=0.3}` for
large ones. The size-dependent *training* schedule (batch size, epochs, steps) is not
searched — it is derived from the train-table size in `fit`.

Heavy deps (PyG / PyTorch Frame / the vendored model) are imported lazily inside
`fit` / `predict`, so importing this module to register the model stays cheap.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from relbench.base import Database, EntityTask, Table, TaskType
from torch.utils.data import DataLoader

from relarena.metrics import get_metric, primary_metric
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace, TaskStats

# Architecture constants — fixed across the paper's runs (expts/*.sh), not tuned.
_CHANNELS = 512
_NUM_HEADS = 4
_NUM_NEIGHBORS = 300  # K: token sequence length (seed + K-1 sampled neighbours)
_NUM_CENTROIDS = 4096  # B: global-attention codebook size
_CONV_TYPE = "full"  # local + global attention
_LR = 1e-4
_WEIGHT_DECAY = 1e-5
_GRAD_CLIP_NORM = 1.0

#: Above this many training nodes the paper fixes L=4 and uses the larger-batch, fewer-
#: epochs schedule; at/below it, the full L×dropout grid and the small-data schedule.
_LARGE_NODE_THRESHOLD = 1_000_000
#: Between this and the large threshold the search space drops the depth sweep and
#: tunes dropout at the default depth only — the full grid's heaviest task blows past
#: practical per-job time limits. Affects the grid only, never the schedule.
_MEDIUM_NODE_THRESHOLD = 100_000
#: Size-derived training schedule `(batch_size, epochs, max_steps_per_epoch)` — from
#: run-large-base-experiments.sh (large) and run-hyperparam-sweep-small-experiments.sh.
_LARGE_SCHEDULE = (1024, 10, 500)
_SMALL_SCHEDULE = (256, 100, 3000)
#: TODO: stopgap. These (dataset, task) refits OOM a single 95 GB GPU at the large-
#: schedule batch (1024) — per-batch attention activations sit at the memory ceiling.
#: Halve their batch until batch size is made memory-aware (or grad checkpointing is
#: added). Keyed on the exact pair, not the task class, since user-churn collides
#: across rel-hm and rel-amazon.
_OOM_HALVE_BATCH_TASKS = {
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-ltv"),
    ("rel-hm", "item-sales"),
    ("rel-hm", "user-churn"),
}

#: The tuned grid: dropout (applied to both ff and attn) × local-transformer depth L.
#: Ordered L=4 first so a budget cap (or the large-task space) keeps the default.
_GRID_DROPOUTS = (0.3, 0.4, 0.5)
_GRID_LAYERS = (4, 1, 8)
_DEFAULT_CONFIG: dict[str, Any] = {"num_layers": 4, "dropout": 0.3}


def _full_grid() -> list[dict[str, Any]]:
    """The 9-config `L × dropout` grid, `{L=4, dropout=0.3}` first."""
    return [
        {"num_layers": layers, "dropout": dropout}
        for layers in _GRID_LAYERS
        for dropout in _GRID_DROPOUTS
    ]


def _medium_grid() -> list[dict[str, Any]]:
    """Dropout sweep at the default depth — a 3-config subset of the full grid."""
    layers = _DEFAULT_CONFIG["num_layers"]
    return [{"num_layers": layers, "dropout": d} for d in _GRID_DROPOUTS]


def relgt_search_space(stats: TaskStats) -> SearchSpace:
    """Size-aware grid: the L×dropout sweep shrinks as the training set grows.

    Above ~1M training nodes a single `{L=4, dropout=0.3}` config; above ~100k, a
    dropout sweep at the default depth; below that, the full L×dropout grid. Each tier
    is a subset of the next-smaller one, cut on the training-set size alone.
    """
    n = stats.num_train_nodes
    if n > _LARGE_NODE_THRESHOLD:
        grid = [dict(_DEFAULT_CONFIG)]
    elif n > _MEDIUM_NODE_THRESHOLD:
        grid = _medium_grid()
    else:
        grid = _full_grid()
    return SearchSpace(fixed_grid=grid, default_overrides=_DEFAULT_CONFIG)


def _schedule(num_train_nodes: int) -> tuple[int, int, int]:
    """`(batch_size, epochs, max_steps_per_epoch)` for the train-set size."""
    return (
        _LARGE_SCHEDULE if num_train_nodes > _LARGE_NODE_THRESHOLD else _SMALL_SCHEDULE
    )


def _forward(model: Any, batch: dict, device: str) -> torch.Tensor:
    """Run `RelGT.forward` on a collated token batch (tensors moved to `device`)."""
    return model(
        neighbor_types=batch["neighbor_types"].to(device),
        node_indices=batch["node_indices"].to(device),
        neighbor_hops=batch["neighbor_hops"].to(device),
        neighbor_times=batch["neighbor_times"].to(device),
        grouped_tf_dict={
            "grouped_tfs": batch["grouped_tfs"],  # TFs moved to device in the encoder
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx": batch["flat_batch_idx"],
            "flat_nbr_idx": batch["flat_nbr_idx"],
        },
        edge_index=batch["edge_index"].to(device),
        batch=batch["batch"].to(device),
    )


def train_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    max_steps: int,
) -> None:
    """One training pass (capped at `max_steps` minibatches)."""
    model.train()
    for steps, batch in enumerate(loader):
        if steps >= max_steps:
            break
        optimizer.zero_grad()
        pred = _forward(model, batch, device).view(-1)
        loss = loss_fn(pred.float(), batch["labels"].to(device).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
        optimizer.step()


@torch.no_grad()
def infer(
    model: Any,
    loader: DataLoader,
    task: EntityTask,
    device: str,
    clamp: tuple[float, float] | None,
) -> np.ndarray:
    """Predict over `loader`, reordered to table-row order via `global_idx`.

    Regression → clamped; binary → `sigmoid`. The loader may be unshuffled, but we
    scatter by each sample's `global_idx` so the returned array always aligns with
    the input table's rows (the contract `EntityTask.evaluate` expects).
    """
    model.eval()
    preds: list[np.ndarray] = []
    idxs: list[np.ndarray] = []
    for batch in loader:
        pred = _forward(model, batch, device)
        if task.task_type == TaskType.REGRESSION:
            assert clamp is not None
            pred = torch.clamp(pred, clamp[0], clamp[1])
        elif task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)
        preds.append(pred.view(-1).cpu().numpy())
        idxs.append(batch["global_idx"].numpy())

    flat = np.concatenate(preds)
    order = np.concatenate(idxs)
    out = np.empty_like(flat)
    out[order] = flat
    return out


@register_model(search_space=relgt_search_space)
class RelGTModel(RelArenaModel):
    """Relational Graph Transformer on the RelBench relational graph.

    Tuned over `relgt_search_space` (L × dropout, size-aware); the default regime
    is `{L=4, dropout=0.3}`. Shares `graphsage`'s graph materialization/cache.
    """

    name = "relgt"

    supported_task_types = frozenset(
        {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.REGRESSION,
        }
    )

    #: RelGT's published protocol trains on train alone and reports the best-val
    #: checkpoint, not a train+val refit. The refit's last-epoch model overtrains
    #: badly on the small, high-epoch tasks (the model peaks on val within a handful
    #: of epochs), so the best-val regime is what reproduces the paper's numbers.
    refit_on_full_data = False

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
        """Build the graph + token cache, then train RelGT (best-checkpoint on val).

        `val_table` is `None` on the refit split — then we train the full fixed
        epoch budget. The training schedule (batch/epochs/steps) is derived from the
        train-table size, matching the paper's per-size regime.
        """
        from torch_geometric.seed import seed_everything

        from relarena.models._shared.gnn.graph import GRAPH_CACHE, build_graph
        from relarena.models._shared.gnn.training import default_device, task_setup
        from relarena.models.relgt._vendor import RelGT
        from relarena.models.relgt.tokenize import MAX_NEIGHBOR_HOP, RelGTTokens

        seed_everything(seed)
        device = default_device()
        data, col_stats_dict = GRAPH_CACHE.get(db, lambda: build_graph(db, device))
        col_names_dict = {nt: data[nt].tf.col_names_dict for nt in data.node_types}

        num_layers = int(self.config["num_layers"])
        dropout = float(self.config["dropout"])
        out_channels, loss_fn, clamp = task_setup(task, train_table)
        batch_size, epochs, max_steps = _schedule(len(train_table.df))
        # task.cache_dir ends in '.../<dataset>/tasks/<task>'; key the OOM stopgap
        # on that exact pair (the task class collides across datasets). Synthetic
        # tasks may lack cache_dir — they are never the OOM offenders, so skip.
        _cache_dir = getattr(task, "cache_dir", None)
        if _cache_dir is not None:
            _parts = Path(_cache_dir).parts
            if len(_parts) >= 3 and (_parts[-3], _parts[-1]) in _OOM_HALVE_BATCH_TASKS:
                batch_size //= 2
        train_tokens = RelGTTokens(
            data,
            task,
            train_table,
            _NUM_NEIGHBORS,
            cache=self.cache,
            run_identity=self.run_identity,
        )
        model = RelGT(
            num_nodes=train_tokens.num_global_nodes,
            max_neighbor_hop=MAX_NEIGHBOR_HOP,
            node_type_map=train_tokens.node_type_map,
            col_names_dict=col_names_dict,
            col_stats_dict=col_stats_dict,
            local_num_layers=num_layers,
            channels=_CHANNELS,
            out_channels=out_channels,
            global_dim=_CHANNELS // 2,
            heads=_NUM_HEADS,
            ff_dropout=dropout,
            attn_dropout=dropout,
            conv_type=_CONV_TYPE,
            num_centroids=_NUM_CENTROIDS,
            sample_node_len=_NUM_NEIGHBORS,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY
        )
        train_loader = DataLoader(
            train_tokens,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_tokens.collate,
        )

        metric_fn = primary_metric(task)
        higher = get_metric(metric_fn).higher_is_better
        best_score = -math.inf if higher else math.inf
        best_state: dict[str, Any] | None = None
        val_loader = None
        if val_table is not None:
            val_tokens = RelGTTokens(
                data,
                task,
                val_table,
                _NUM_NEIGHBORS,
                cache=self.cache,
                run_identity=self.run_identity,
            )
            val_loader = DataLoader(
                val_tokens,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=val_tokens.collate,
            )
            val_true = val_table.df[task.target_col].to_numpy()

        start = time.monotonic()
        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, loss_fn, device, max_steps)
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
        self._batch_size = batch_size

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Run RelGT over `table`; return the evaluate-contract array."""
        from relarena.models._shared.gnn.graph import GRAPH_CACHE, build_graph
        from relarena.models.relgt.tokenize import RelGTTokens

        data, _ = GRAPH_CACHE.get(db, lambda: build_graph(db, self._device))
        tokens = RelGTTokens(
            data,
            task,
            table,
            _NUM_NEIGHBORS,
            cache=self.cache,
            run_identity=self.run_identity,
        )
        loader = DataLoader(
            tokens,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=tokens.collate,
        )
        return infer(self._model, loader, task, self._device, self._clamp)
