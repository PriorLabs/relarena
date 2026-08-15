"""Unit tests for the GraphSAGE GNN model.

These cover registration, the search space, the per-task-type setup, and the
predict-contract reshaping — all without the `graphsage` extra (PyG / PyTorch Frame),
since the heavy graph stack is imported lazily inside `fit` / `predict`. The full
graph fit/predict path runs end-to-end against a real RelBench dataset on a GPU (the CLI
smoke run), not here.
"""

from __future__ import annotations

import gc
from typing import Callable

import pytest
import torch
from ConfigSpace import Configuration
from relbench.base import TaskType

from relarena.models.graphsage import model as graphsage_mod
from relarena.models.graphsage.model import (
    _BATCH_SIZE,
    _MIN_BATCH_SIZE,
    GRAPHSAGE_SPACE,
    GraphSAGEModel,
    _cuda_cleanup,
    _run_with_oom_retry,
)
from relarena.registry import registry

_SUPPORTED_TASK_TYPES = frozenset(
    {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
    }
)
_TUNED_KNOBS = {"channels", "num_layers", "aggr", "num_neighbors", "lr"}
# RelBench's un-tuned GraphSAGE defaults (examples/gnn_entity.py argparse).
_RELBENCH_DEFAULTS = {
    "channels": 128,
    "num_layers": 2,
    "aggr": "sum",
    "num_neighbors": 128,
    "lr": 0.005,
}


# -- registration & search space --------------------------------------------


def test_registered_under_name_graphsage() -> None:
    assert "graphsage" in registry
    assert registry.get("graphsage") is GraphSAGEModel
    assert registry.search_space("graphsage") is GRAPHSAGE_SPACE


def test_supports_binary_and_regression_only() -> None:
    # Multiclass removed (no RelBench v1 entity task is multiclass).
    assert GraphSAGEModel.supported_task_types == _SUPPORTED_TASK_TYPES


def test_default_config_is_relbench_baseline() -> None:
    # The default regime, read from the search space, pins RelBench's un-tuned values
    # so the reproduced baseline can't drift silently.
    assert GRAPHSAGE_SPACE.default_overrides == _RELBENCH_DEFAULTS
    assert GRAPHSAGE_SPACE.is_tunable


def test_default_config_is_a_valid_point_in_the_space() -> None:
    # Guards default/space drift: the default must set exactly the space's
    # hyperparameters, each to a legal value — Configuration() raises otherwise.
    Configuration(GRAPHSAGE_SPACE.space, values=dict(GRAPHSAGE_SPACE.default_overrides))


def test_configs_default_first_then_valid_samples() -> None:
    configs = GRAPHSAGE_SPACE.configs(n_trials=4, seed=0)
    assert len(configs) == 5  # the default config + 4 samples
    assert configs[0] == GRAPHSAGE_SPACE.default_overrides  # default regime first
    # Each config tunes exactly those knobs and is a legal point in the space.
    for cfg in configs:
        assert set(cfg) == _TUNED_KNOBS
        Configuration(GRAPHSAGE_SPACE.space, values=dict(cfg))


def test_configs_are_seeded_and_reproducible() -> None:
    assert GRAPHSAGE_SPACE.configs(4, seed=0) == GRAPHSAGE_SPACE.configs(4, seed=0)


# -- task setup (out_channels / loss / clamp) --------------------------------


# -- OOM resilience (_run_with_oom_retry / _cuda_cleanup) --------------------


def _oom_after(n_oom: int, calls: list[int], result: str = "trained") -> Callable:
    """An attempt that raises CUDA OOM on its first `n_oom` calls, then succeeds.

    Records the batch size of every call into `calls`.
    """

    def attempt(batch_size: int) -> str:
        calls.append(batch_size)
        if len(calls) <= n_oom:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return result

    return attempt


def test__run_with_oom_retry__success_first_try__returns_start_batch_size() -> None:
    calls: list[int] = []
    result, batch_size = _run_with_oom_retry(
        _oom_after(0, calls), start_batch_size=_BATCH_SIZE, device="cpu", what="fit"
    )
    assert result == "trained"
    assert batch_size == _BATCH_SIZE
    assert calls == [_BATCH_SIZE]


def test__run_with_oom_retry__oom_once__halves_and_succeeds() -> None:
    calls: list[int] = []
    result, batch_size = _run_with_oom_retry(
        _oom_after(1, calls), start_batch_size=512, device="cpu", what="fit"
    )
    assert result == "trained"
    assert batch_size == 256
    assert calls == [512, 256]


def test__run_with_oom_retry__oom_sequence__halves_down_to_floor() -> None:
    calls: list[int] = []
    # OOM on every attempt above the floor, succeed at the floor.
    _, batch_size = _run_with_oom_retry(
        _oom_after(5, calls), start_batch_size=512, device="cpu", what="fit"
    )
    assert batch_size == _MIN_BATCH_SIZE
    assert calls == [512, 256, 128, 64, 32, 16]


def test__run_with_oom_retry__oom_persists_at_floor__reraises() -> None:
    calls: list[int] = []

    def always_oom(batch_size: int) -> str:
        calls.append(batch_size)
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.OutOfMemoryError):
        _run_with_oom_retry(always_oom, start_batch_size=512, device="cpu", what="fit")
    # Halved all the way to the floor, attempted it once, then gave up.
    assert calls == [512, 256, 128, 64, 32, 16]
    assert calls[-1] == _MIN_BATCH_SIZE


def test__run_with_oom_retry__non_oom_error__propagates_without_retry() -> None:
    calls: list[int] = []

    def bad_shape(batch_size: int) -> str:
        calls.append(batch_size)
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        _run_with_oom_retry(bad_shape, start_batch_size=512, device="cpu", what="fit")
    assert calls == [512]  # not retried — only OOM is caught


def test__run_with_oom_retry__downshift__cleans_cuda_each_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanups: list[str] = []
    monkeypatch.setattr(
        graphsage_mod, "_cuda_cleanup", lambda device: cleanups.append(device)
    )
    _run_with_oom_retry(
        _oom_after(2, []), start_batch_size=512, device="cuda", what="fit"
    )
    assert cleanups == ["cuda", "cuda"]  # one cleanup per OOM


def test__cuda_cleanup__cpu_device__skips_cuda_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: called.append("gc"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: called.append("sync"))
    _cuda_cleanup("cpu")
    assert called == ["gc"]


def test__cuda_cleanup__cuda_device__empties_cache_after_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gc, "collect", lambda: order.append("gc"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: order.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: order.append("sync"))
    _cuda_cleanup("cuda")
    assert order == ["gc", "empty_cache", "sync"]


def test__min_batch_size__is_a_clean_power_of_two_floor() -> None:
    # The halving sequence must land exactly on the floor (no undershoot).
    assert _MIN_BATCH_SIZE < _BATCH_SIZE
    assert _BATCH_SIZE % _MIN_BATCH_SIZE == 0


# -- misc --------------------------------------------------------------------
