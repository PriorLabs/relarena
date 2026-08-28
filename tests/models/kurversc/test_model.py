from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import kurversc
import numpy as np
import pandas as pd
from pytest import MonkeyPatch
from relbench.base import Table, TaskType

from relarena.identity import RunIdentity
from relarena.models.kurversc import KURVERSC_SPACE, KurveRSCSystem
from relarena.tasks import RELBENCH_V1_DATASETS, list_entity_tasks


def _label_table(split: str) -> Table:
    return Table(
        df=pd.DataFrame(
            {
                "uid": [0, 1, 2, 3],
                "timestamp": pd.Timestamp("2020-01-01"),
                "target": [0, 1, 0, 1],
                "split": split,
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="timestamp",
    )


def test__kurversc__registration_and_system_contract() -> None:
    import relarena.models  # noqa: F401
    from relarena.registry import registry

    assert registry.get("kurversc") is KurveRSCSystem
    assert KURVERSC_SPACE.default_overrides == {
        "full_training_frames": 1,
        "sample_rows": 100_000,
        "feature_family_max_columns": 4,
    }
    assert KurveRSCSystem.kind == "system"
    assert KurveRSCSystem.refit_on_full_data is True


def test__kurversc__supports_all_relbench_v1_entity_tasks() -> None:
    specs = list_entity_tasks(RELBENCH_V1_DATASETS)

    assert len(specs) == 21
    assert all(spec.task_type in KurveRSCSystem.supported_task_types for spec in specs)


def test__fit_predict__delegates_only_through_public_kurversc_api(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    config = kurversc.GraphConfig(depth=2)
    problem = SimpleNamespace(
        parent_node="parent",
        tables=("table",),
        relationships=("relationship",),
        fit_kwargs=lambda: {"parent_node": "parent", "label_node": "labels"},
    )

    def fake_problem(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["problem"] = (args, kwargs)
        return problem

    def fake_fit(**kwargs: Any) -> SimpleNamespace:
        calls["fit"] = kwargs
        return SimpleNamespace(best_config=config)

    def fake_predict(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["predict"] = (args, kwargs)
        return pd.DataFrame({"prediction": [0.1, 0.2, 0.3, 0.4]})

    monkeypatch.setattr(kurversc, "relbench_problem_from_objects", fake_problem)
    monkeypatch.setattr(kurversc, "fit", fake_fit)
    monkeypatch.setattr(kurversc, "predict", fake_predict)
    task = SimpleNamespace(
        entity_col="uid",
        entity_table="users",
        target_col="target",
        time_col="timestamp",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    identity = RunIdentity("rel-test", "db", "task", "labels", phase="inner")
    model = KurveRSCSystem(
        {
            "full_training_frames": 1,
            "sample_rows": 10_000,
            "feature_family_max_columns": 3,
        },
        run_identity=identity,
    )
    train = _label_table("train")
    validation = _label_table("validation")

    model.fit(task, SimpleNamespace(), train, validation, seed=7)
    prediction = model.predict(task, SimpleNamespace(), validation)

    assert np.allclose(prediction, [0.1, 0.2, 0.3, 0.4])
    assert calls["problem"][1]["sample_rows"] == 10_000
    assert calls["fit"]["sample_rows"] == 10_000
    assert calls["fit"]["feature_family_max_columns"] == 3
    assert calls["fit"]["random_state"] == 7
    assert calls["predict"][1]["use_validation_model"] is True


def test__fit__rejects_too_small_search_sample() -> None:
    model = KurveRSCSystem({"sample_rows": 1})

    with np.testing.assert_raises_regex(ValueError, "sample_rows must be at least 2"):
        model.fit(
            SimpleNamespace(),
            SimpleNamespace(),
            _label_table("train"),
            _label_table("validation"),
            seed=7,
        )


def test__fit__rejects_nonpositive_feature_family_cap() -> None:
    model = KurveRSCSystem({"feature_family_max_columns": 0})

    with np.testing.assert_raises_regex(
        ValueError, "feature_family_max_columns must be positive or null"
    ):
        model.fit(
            SimpleNamespace(),
            SimpleNamespace(),
            _label_table("train"),
            _label_table("validation"),
            seed=7,
        )


def test__outer_refit__reuses_inner_graph_config(monkeypatch: MonkeyPatch) -> None:
    fit_calls: list[dict[str, Any]] = []
    config = kurversc.GraphConfig(depth=2)
    problem = SimpleNamespace(
        parent_node="parent",
        tables=(),
        relationships=(),
        fit_kwargs=lambda: {"parent_node": "parent", "label_node": "labels"},
    )

    monkeypatch.setattr(
        kurversc,
        "relbench_problem_from_objects",
        lambda *args, **kwargs: problem,
    )

    def fake_fit(**kwargs: Any) -> SimpleNamespace:
        fit_calls.append(kwargs)
        return SimpleNamespace(best_config=config)

    monkeypatch.setattr(kurversc, "fit", fake_fit)
    task = SimpleNamespace(
        entity_col="uid",
        time_col="timestamp",
        dataset=SimpleNamespace(val_timestamp=pd.Timestamp("2020-02-01")),
    )
    base_identity = RunIdentity("rel-outer", "db", "badge", "task")
    train = _label_table("train")
    train.df["timestamp"] = pd.Timestamp("2020-01-01")
    validation = _label_table("validation")
    validation.df["timestamp"] = pd.Timestamp("2020-03-01")

    inner = KurveRSCSystem(
        {"full_training_frames": 1},
        run_identity=base_identity.for_phase("inner"),
    )
    inner.fit(task, SimpleNamespace(), train, validation, seed=11)
    combined = _label_table("combined")
    combined.df = pd.concat([train.df, validation.df], ignore_index=True)
    outer = KurveRSCSystem(
        {"full_training_frames": 1},
        run_identity=base_identity.for_phase("outer"),
    )
    outer.fit(task, SimpleNamespace(), combined, None, seed=11)

    assert "graph_configs" not in fit_calls[0]
    assert fit_calls[1]["graph_configs"] == (config,)
    assert outer._use_validation_model is False
