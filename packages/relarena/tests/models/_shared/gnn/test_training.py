"""Tests for the GNN training / inference loop shared by the GNN baselines.

`task_setup` and `infer` are exercised here rather than under a model, because all
three GNN adapters route through them; the graphsage tests keep only what is
graphsage's own (OOM retry, batch-size floor, its search space).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from relbench.base import TaskType

from relarena.models._shared.gnn.training import default_device, infer, task_setup


def test_task_setup_binary() -> None:
    task = SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION, target_col="y")
    out_channels, loss_fn, clamp = task_setup(task, train_table=None)
    assert out_channels == 1
    assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
    assert clamp is None


def test_task_setup_rejects_multiclass() -> None:
    # Multiclass support was removed; _task_setup must reject it.
    task = SimpleNamespace(task_type=TaskType.MULTICLASS_CLASSIFICATION, target_col="y")
    with pytest.raises(ValueError, match="does not support task type"):
        task_setup(task, train_table=None)


def test_task_setup_regression_clamps_to_train_percentiles() -> None:
    task = SimpleNamespace(task_type=TaskType.REGRESSION, target_col="y")
    y = np.arange(0, 101, dtype=float)  # 0..100
    train_table = SimpleNamespace(df=pd.DataFrame({"y": y}))
    out_channels, loss_fn, clamp = task_setup(task, train_table)
    assert out_channels == 1
    assert isinstance(loss_fn, torch.nn.L1Loss)
    assert clamp == (float(np.percentile(y, 2)), float(np.percentile(y, 98)))


def test_task_setup_rejects_unsupported_task_type() -> None:
    task = SimpleNamespace(task_type=TaskType.MULTILABEL_CLASSIFICATION, target_col="y")
    with pytest.raises(ValueError, match="does not support task type"):
        task_setup(task, train_table=None)


# -- predict-contract reshaping (_infer) -------------------------------------


class _StubModel(torch.nn.Module):
    """A model that ignores the batch and returns preset logits per call."""

    def __init__(self, logits_per_batch: list[torch.Tensor]) -> None:
        super().__init__()
        self._logits = logits_per_batch
        self._i = 0

    def forward(self, batch: object, entity_table: str) -> torch.Tensor:
        out = self._logits[self._i]
        self._i += 1
        return out


class _StubBatch:
    """A minimal batch: `.to(device)` is a no-op returning itself."""

    def to(self, device: object) -> "_StubBatch":
        return self


def _stub_loader(n_batches: int) -> list[_StubBatch]:
    return [_StubBatch() for _ in range(n_batches)]


def test_infer_regression_clamps_and_flattens() -> None:
    task = SimpleNamespace(task_type=TaskType.REGRESSION, entity_table="e")
    logits = [torch.tensor([[-5.0], [0.5], [5.0]])]
    out = infer(_StubModel(logits), _stub_loader(1), task, "cpu", clamp=(0.0, 1.0))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_infer_binary_applies_sigmoid_and_flattens() -> None:
    task = SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION, entity_table="e")
    logits = [torch.tensor([[0.0], [2.0]])]
    out = infer(_StubModel(logits), _stub_loader(1), task, "cpu", clamp=None)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, torch.sigmoid(logits[0].view(-1)).numpy())


def test_infer_concatenates_batches_in_loader_order() -> None:
    # shuffle=False -> loader order == table row order; predictions must concatenate
    # in that order (so they align with EntityTask.evaluate's expected rows).
    task = SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION, entity_table="e")
    logits = [torch.tensor([[-10.0]]), torch.tensor([[10.0]])]
    out = infer(_StubModel(logits), _stub_loader(2), task, "cpu", clamp=None)
    assert out.shape == (2,)
    assert out[0] < 0.5 < out[1]


def test_default_device_is_cpu_or_cuda() -> None:
    assert default_device() in ("cpu", "cuda")
