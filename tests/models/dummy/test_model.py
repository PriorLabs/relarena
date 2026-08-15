"""Unit tests for the constant (optimal-constant) baselines.

Uses lightweight stand-in objects for task/table so the tests run without a
RelBench data download — the models only touch `task.{target_col,task_type,
metrics,entity_col}` and `table.df`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from relbench.base import TaskType

from relarena.models.dummy import DummyBaseline, DummyPerEntityBaseline


def _task(
    task_type: TaskType,
    metric_name: str,
    target_col: str = "y",
    entity_col: str = "eid",
) -> SimpleNamespace:
    # primary_metric(task) returns task.metrics[0]; the model reads its __name__.
    metric = SimpleNamespace(__name__=metric_name)
    return SimpleNamespace(
        task_type=task_type,
        target_col=target_col,
        metrics=[metric],
        entity_col=entity_col,
    )


def _table(values: list, col: str = "y") -> SimpleNamespace:
    return SimpleNamespace(df=pd.DataFrame({col: list(values)}))


def _entity_table(entities: list, values: list | None = None) -> SimpleNamespace:
    data = {"eid": list(entities)}
    if values is not None:
        data["y"] = list(values)
    return SimpleNamespace(df=pd.DataFrame(data))


def _fit(task: SimpleNamespace, train_values: list) -> DummyBaseline:
    m = DummyBaseline({})
    m.fit(task, None, _table(train_values), None, seed=0)
    return m


def test_regression_predicts_median() -> None:
    # The regression primary metric is MAE, whose optimal constant is the median.
    m = _fit(_task(TaskType.REGRESSION, "mae"), [1.0, 2.0, 100.0])  # median 2, mean ~34
    pred = m.predict(None, None, _table([0, 0, 0, 0]))
    assert pred.shape == (4,)
    assert np.allclose(pred, 2.0)


def test_binary_predicts_positive_rate() -> None:
    m = _fit(_task(TaskType.BINARY_CLASSIFICATION, "roc_auc"), [0, 0, 0, 1])  # p = 0.25
    pred = m.predict(None, None, _table([0, 0, 0]))
    assert pred.shape == (3,)
    assert np.allclose(pred, 0.25)


def test_registered_under_name_constant_global() -> None:
    import relarena.models  # noqa: F401  (triggers registration)
    from relarena.registry import registry

    assert "constant-global" in registry
    assert registry.get("constant-global") is DummyBaseline


def _fit_per_entity(
    task: SimpleNamespace, entities: list, values: list
) -> DummyPerEntityBaseline:
    m = DummyPerEntityBaseline({})
    m.fit(task, None, _entity_table(entities, values), None, seed=0)
    return m


def test_per_entity_regression_predicts_each_entitys_median() -> None:
    # MAE -> per-entity median; entity 0 -> median(1, 3) = 2, entity 1 -> 10.
    m = _fit_per_entity(_task(TaskType.REGRESSION, "mae"), [0, 0, 1], [1.0, 3.0, 10.0])
    pred = m.predict(None, None, _entity_table([1, 0]))
    assert np.allclose(pred, [10.0, 2.0])


def test_per_entity_binary_predicts_each_entitys_positive_rate() -> None:
    # entity 0 -> mean(0, 1) = 0.5, entity 1 -> mean(1, 1) = 1.0.
    m = _fit_per_entity(
        _task(TaskType.BINARY_CLASSIFICATION, "roc_auc"), [0, 0, 1, 1], [0, 1, 1, 1]
    )
    pred = m.predict(None, None, _entity_table([0, 1]))
    assert np.allclose(pred, [0.5, 1.0])


def test_per_entity_unseen_entity_falls_back_to_global() -> None:
    # Global median over [1, 3, 10] is 3; entity 99 is absent from training.
    m = _fit_per_entity(_task(TaskType.REGRESSION, "mae"), [0, 0, 1], [1.0, 3.0, 10.0])
    pred = m.predict(None, None, _entity_table([99, 0]))
    assert np.allclose(pred, [3.0, 2.0])


def test_per_entity_registered_under_name_constant_per_entity() -> None:
    import relarena.models  # noqa: F401  (triggers registration)
    from relarena.registry import registry

    assert "constant-per-entity" in registry
    assert registry.get("constant-per-entity") is DummyPerEntityBaseline
