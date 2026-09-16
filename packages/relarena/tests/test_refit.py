"""Final benchmark refit and evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from relbench.base import Table, TaskType

from relarena.refit import refit_and_evaluate
from relarena_core.cache import CacheConfig
from relarena_core.dataset import OuterSplit
from relarena_core.identity import RunIdentity
from relarena_core.model import RelArenaModel


def _outer_table(entities: list[int], times: list[int], ys: list[float]) -> Table:
    return Table(
        df=pd.DataFrame({"entity": entities, "t": times, "y": ys}),
        fkey_col_to_pkey_table={"entity": "e"},
        pkey_col=None,
        time_col="t",
    )


def _outer_split() -> OuterSplit:
    return OuterSplit(
        db_state=SimpleNamespace(),
        cutoff=pd.Timestamp("2020-01-01"),
        train_table=_outer_table([1, 2], [10, 11], [0.0, 1.0]),
        val_table=_outer_table([3], [12], [2.0]),
        eval_table=_outer_table([4, 5], [13, 14], [3.0, 4.0]),
    )


def _stub_task() -> Any:
    # primary_metric reads task_type; evaluate_predictions reads
    # task.metrics + task.evaluate.
    return SimpleNamespace(
        task_type=TaskType.REGRESSION,
        metrics=[],
        evaluate=lambda pred, target, metrics=None: {"mae": 0.5},
    )


def _capturing_model(refit_full: bool) -> tuple[type[RelArenaModel], dict]:
    captured: dict = {}

    class _M(RelArenaModel):
        name = "capture"
        refit_on_full_data = refit_full

        def fit(self, task, db, train_table, val_table, *, seed, time_limit=None):  # noqa: ANN001, ANN202
            captured["train_y"] = list(train_table.df["y"])
            captured["val_y"] = None if val_table is None else list(val_table.df["y"])
            captured["cache"] = self.cache
            captured["run_identity"] = self.run_identity

        def predict(self, task, db, table) -> np.ndarray:  # noqa: ANN001
            return np.zeros(len(table.df))

    return _M, captured


def test__refit_and_evaluate__full_data__fits_on_train_plus_val_no_monitor() -> None:
    model_cls, captured = _capturing_model(refit_full=True)
    out = refit_and_evaluate(model_cls, {}, _stub_task(), _outer_split(), seed=0)
    assert captured["train_y"] == [0.0, 1.0, 2.0]  # train + val union
    assert captured["val_y"] is None  # nothing held out to monitor
    assert out["test_score"] == 0.5


def test__refit_and_evaluate__best_val__fits_on_train_only_with_val_monitor() -> None:
    model_cls, captured = _capturing_model(refit_full=False)
    out = refit_and_evaluate(model_cls, {}, _stub_task(), _outer_split(), seed=0)
    assert captured["train_y"] == [0.0, 1.0]  # train only
    assert captured["val_y"] == [2.0]  # val passed through as the monitor set
    assert out["test_score"] == 0.5


def test__refit_and_evaluate__cache_config__reaches_model(tmp_path: Path) -> None:
    model_cls, captured = _capturing_model(refit_full=True)
    cache = CacheConfig(tmp_path, "raise")
    refit_and_evaluate(model_cls, {}, _stub_task(), _outer_split(), seed=0, cache=cache)
    assert captured["cache"] is cache


def test__refit_and_evaluate__run_identity__reaches_model() -> None:
    model_cls, captured = _capturing_model(refit_full=True)
    identity = RunIdentity("dataset", "db", "task", "labels", phase="outer")
    refit_and_evaluate(
        model_cls,
        {},
        _stub_task(),
        _outer_split(),
        seed=0,
        run_identity=identity,
    )
    assert captured["run_identity"] is identity
