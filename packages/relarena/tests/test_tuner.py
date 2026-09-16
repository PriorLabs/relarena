"""Tests for tuner helpers and the search-space config plan."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace, Integer
from relbench.base import Table, TaskType

from relarena.cache import CacheConfig
from relarena.dataset import OuterSplit, concat_tables
from relarena.identity import RunIdentity
from relarena.model import RelArenaModel
from relarena.search_space import SearchSpace
from relarena.tuner import _concise_error, plan_configs, refit_and_evaluate


def _random_space() -> SearchSpace:
    return SearchSpace(
        space=ConfigurationSpace(space=[Integer("x", (1, 100))], seed=0),
        default_overrides={},
    )


def _grid_space() -> SearchSpace:
    return SearchSpace(
        fixed_grid=[{"d": 3}, {"d": 2}, {"d": 1}], default_overrides={"d": 2}
    )


def test_plan_configs_random_default_plus_samples() -> None:
    plan = plan_configs(_random_space(), n_trials=3, seed=0)
    tags = [t for t, _ in plan]
    assert tags[0] == "default" and plan[0][1] == {}  # the empty default comes first
    assert len(plan) == 4  # default + 3 random samples
    assert all("x" in cfg for _, cfg in plan[1:])


def test_plan_configs_grid_uses_grid_in_order() -> None:
    plan = plan_configs(_grid_space(), n_trials=99, seed=0)
    configs = [c for _, c in plan]
    assert configs == [{"d": 3}, {"d": 2}, {"d": 1}]  # whole grid, deepest first
    assert plan[1] == ("default", {"d": 2})  # default-matching entry tagged "default"


def test_plan_configs_grid_capped_at_n_trials_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        plan = plan_configs(_grid_space(), n_trials=2, seed=0)
    configs = [c for _, c in plan]
    # budget < grid -> keep the first n_trials (the deepest-first grid keeps d=3, d=2)
    assert configs == [{"d": 3}, {"d": 2}]
    # default still tagged when it survives the cap
    assert plan[1] == ("default", {"d": 2})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("dropping 1" in m for m in warnings)


def test_plan_configs_grid_within_budget_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        plan_configs(_grid_space(), n_trials=3, seed=0)  # exactly fits
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_default_overrides_not_in_fixed_grid_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        SearchSpace(fixed_grid=[{"d": 3}, {"d": 1}], default_overrides={"d": 2})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not in the fixed_grid" in m for m in warnings)


def test_default_overrides_in_fixed_grid_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        SearchSpace(fixed_grid=[{"d": 3}, {"d": 2}], default_overrides={"d": 2})
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test__concise_error__one_line_type_message_and_raise_site() -> None:
    try:
        raise ValueError("bad\nstuff")  # multi-line message must collapse to one line
    except ValueError as exc:
        summary = _concise_error(exc)
    assert "\n" not in summary
    assert summary.startswith("ValueError: bad stuff (")
    assert "test_tuner.py:" in summary  # innermost frame = where it was raised


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
    # primary_metric reads task_type; _evaluate reads task.metrics + task.evaluate.
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


def test_concat_tables_unions_rows_and_keeps_schema() -> None:
    a = Table(
        df=pd.DataFrame({"entity": [1, 2], "t": [10, 11], "y": [0.0, 1.0]}),
        fkey_col_to_pkey_table={"entity": "users"},
        pkey_col=None,
        time_col="t",
    )
    b = Table(
        df=pd.DataFrame({"entity": [3], "t": [12], "y": [2.0]}),
        fkey_col_to_pkey_table={"entity": "users"},
        pkey_col=None,
        time_col="t",
    )
    c = concat_tables(a, b)
    assert len(c.df) == 3
    assert list(c.df["y"]) == [0.0, 1.0, 2.0]
    assert c.time_col == "t"
    assert c.fkey_col_to_pkey_table == {"entity": "users"}
    # inputs are untouched
    assert len(a.df) == 2 and len(b.df) == 1
