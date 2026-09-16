from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import kurversc
import numpy as np
import pandas as pd
import pytest
from relbench.base import Table, TaskType

from relarena.identity import RunIdentity
from relarena.models.kurversc import KURVERSC_DEFAULTS, KurveRSCSystem
from relarena.registry import registry
from relarena.tasks import RELBENCH_V1_DATASETS, list_entity_tasks


def _label_table(*, masked: bool = False) -> Table:
    values: dict[str, Any] = {
        "uid": [0, 1, 2, 3],
        "timestamp": [pd.Timestamp("2020-01-01")] * 4,
    }
    if not masked:
        values["target"] = [0, 1, 0, 1]
    return Table(
        df=pd.DataFrame(values),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="timestamp",
    )


def test__kurversc__is_registered_as_a_native_system() -> None:
    import relarena.models  # noqa: F401

    assert registry.get("kurversc") is KurveRSCSystem
    assert registry.kind("kurversc") == "system"
    with pytest.raises(TypeError, match="no harness search space"):
        registry.search_space("kurversc")


def test__kurversc__report_facing_defaults_are_explicit() -> None:
    assert KURVERSC_DEFAULTS == {
        "full_training_frames": 1,
        "search_training_frames": 1,
        "sample_rows": 50_000,
        "screening_rows": 10_000,
        "confirmation_top_k": 8,
        "search_full_data": True,
        "rerank_top_k": 3,
        "rerank_cutoff_frames": 3,
        "rerank_stability_penalty": 0.25,
        "feature_family_max_columns": 4,
        "adaptive_depth_promotion": True,
        "capability_pruning": True,
        "search_max_features": 8_000,
        "infer_ts_periods": False,
        "auto_text_features": False,
        "auto_annotate_max_text_columns": None,
        "duckdb_memory_limit": "64GB",
        "duckdb_max_temp_directory_size": "128GB",
    }


def test__kurversc__supports_all_relbench_v1_entity_tasks() -> None:
    specs = list_entity_tasks(RELBENCH_V1_DATASETS)

    assert len(specs) == 21
    assert all(spec.task_type in KurveRSCSystem.supported_task_types for spec in specs)


def test__run__carries_frozen_inner_graph_into_outer_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[Any]] = {"problem": [], "fit": [], "predict": []}
    selected_config = kurversc.GraphConfig(depth=2)
    frozen_plan = {"records": [{"method_name": "auto_features"}]}

    def fake_problem(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["problem"].append((args, kwargs))
        return SimpleNamespace(
            parent_node="parent",
            tables=("table",),
            relationships=("relationship",),
            fit_kwargs=lambda: {"parent_node": "parent", "label_node": "labels"},
        )

    def fake_fit(**kwargs: Any) -> SimpleNamespace:
        calls["fit"].append(kwargs)
        return SimpleNamespace(
            recommended_config=selected_config,
            execution_plan=frozen_plan,
        )

    def fake_predict(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["predict"].append((args, kwargs))
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
    train = _label_table()
    validation = _label_table()
    test = _label_table(masked=True)
    inner = SimpleNamespace(
        db_state="inner-db",
        train_table=train,
        eval_table=validation,
    )
    outer = SimpleNamespace(
        db_state="outer-db",
        train_table=train,
        val_table=validation,
        eval_table=test,
    )
    system = KurveRSCSystem(
        run_identity=RunIdentity("rel-test", "db", "task", "labels")
    )

    prediction = system.run(task, inner_split=inner, outer_split=outer, seed=7)

    assert np.allclose(prediction, [0.1, 0.2, 0.3, 0.4])
    assert len(calls["problem"]) == 2
    assert calls["problem"][0][0][1] == "inner-db"
    assert calls["problem"][1][0][1] == "outer-db"
    assert calls["problem"][0][1]["max_train_timestamps"] == 3
    assert calls["problem"][1][1]["max_train_timestamps"] == 1
    assert calls["problem"][0][1]["dataset_name"] == "rel-test"
    assert calls["problem"][0][1]["task_name"] == "task"

    inner_fit, outer_fit = calls["fit"]
    assert "preselected_config" not in inner_fit
    assert "preselected_execution_plan" not in inner_fit
    assert outer_fit["preselected_config"] == selected_config
    assert outer_fit["preselected_execution_plan"] == frozen_plan
    assert outer_fit["preselected_execution_plan"] is not frozen_plan
    assert outer_fit["random_state"] == 7

    predict_kwargs = calls["predict"][0][1]
    assert predict_kwargs["use_validation_model"] is False
    assert predict_kwargs["duckdb_memory_limit"] == "64GB"
    assert predict_kwargs["duckdb_max_temp_directory_size"] == "128GB"


def test__fit_arm__outer_requires_inner_selection() -> None:
    system = KurveRSCSystem()

    with pytest.raises(RuntimeError, match="requires the inner selection"):
        system._fit_arm(
            SimpleNamespace(),
            SimpleNamespace(),
            _label_table(),
            _label_table(),
            seed=0,
            phase="outer",
        )


def test__system__does_not_expose_model_fit_predict_contract() -> None:
    assert not hasattr(KurveRSCSystem, "fit")
    assert not hasattr(KurveRSCSystem, "predict")
