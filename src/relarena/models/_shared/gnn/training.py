"""The training and inference loop shared by the GNN baselines.

`graphsage`, `relgnn` and `relgt` differ in their message passing, not in how a
task is turned into a loss, how an epoch is stepped, or how predictions are shaped
for `EntityTask.evaluate`. Those pieces live here so the three adapters do not
import them from each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from relbench.base import EntityTask, Table, TaskType
from torch.nn import BCEWithLogitsLoss, L1Loss

#: Percentiles used to clamp regression predictions to the training range.
REGRESSION_CLAMP_PERCENTILES = (2, 98)


def default_device() -> str:
    """Return `"cuda"` if a GPU is visible to torch, else `"cpu"`."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def task_setup(
    task: EntityTask, train_table: Table
) -> tuple[int, torch.nn.Module, tuple[float, float] | None]:
    """`(out_channels, loss_fn, clamp)` per task type; clamp = regression only."""
    task_type = task.task_type
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return 1, BCEWithLogitsLoss(), None
    if task_type == TaskType.REGRESSION:
        y = train_table.df[task.target_col].to_numpy(dtype=float)
        clamp_min, clamp_max = np.percentile(y, REGRESSION_CLAMP_PERCENTILES)
        return 1, L1Loss(), (float(clamp_min), float(clamp_max))
    # Multiclass intentionally unsupported: no RelBench v1 entity task is multiclass,
    # so the path was untested; re-add (out_channels=num_classes, CrossEntropyLoss,
    # softmax) if a multiclass task appears.
    raise ValueError(f"This GNN baseline does not support task type {task_type}.")


def train_epoch(
    model: Any,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    task: EntityTask,
    device: str,
    max_steps: int,
) -> None:
    """One training pass (capped at `max_steps` minibatches)."""
    model.train()
    entity = task.entity_table
    for steps, batch in enumerate(loader):
        if steps >= max_steps:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, entity).view(-1)
        loss = loss_fn(pred.float(), batch[entity].y.float())
        loss.backward()
        optimizer.step()


@torch.no_grad()
def infer(
    model: Any,
    loader: Any,
    task: EntityTask,
    device: str,
    clamp: tuple[float, float] | None,
) -> np.ndarray:
    """Predict over `loader` and shape to the evaluate contract.

    Regression -> clamped `(N,)`; binary -> `sigmoid` `(N,)` = P(positive). Loader
    order (`shuffle=False`) preserves row alignment with the input table.
    """
    model.eval()
    preds: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, task.entity_table)
        if task.task_type == TaskType.REGRESSION:
            assert clamp is not None
            pred = torch.clamp(pred, clamp[0], clamp[1])
        elif task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)
        preds.append(pred.view(-1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)
