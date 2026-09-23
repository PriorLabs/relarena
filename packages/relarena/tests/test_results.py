"""Benchmark result tables."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.evaluation import to_bencheval_frame
from relarena.results import summary_to_dataframe, trials_to_dataframe
from relarena.runner import ExperimentSummary
from relarena_core.results import TrialResult, config_id_for


def test_trials_to_dataframe_drops_arrays() -> None:
    t = TrialResult(
        config={"a": 1},
        config_id=config_id_for({"a": 1}),
        config_tag="default",
        val_score=0.5,
        val_pred=np.zeros(3),
    )
    df = trials_to_dataframe([t])
    assert "val_pred" not in df.columns
    assert df.loc[0, "val_score"] == 0.5


def test_trials_to_dataframe_flattens_metrics() -> None:
    t = TrialResult(
        config={},
        config_id=config_id_for({}),
        config_tag="default",
        val_score=0.5,
        test_score=0.6,
        val_metrics={"mae": 0.5, "rmse": 0.7},
        test_metrics={"mae": 0.6, "rmse": 0.8},
    )
    df = trials_to_dataframe([t])
    # the metric dicts themselves are not columns; their entries are flattened.
    assert "val_metrics" not in df.columns and "test_metrics" not in df.columns
    assert df.loc[0, "val_mae"] == 0.5 and df.loc[0, "val_rmse"] == 0.7
    assert df.loc[0, "test_mae"] == 0.6 and df.loc[0, "test_rmse"] == 0.8


@pytest.mark.parametrize("tags", [["default"], ["default", "r1"], ["r0"]])
def test_diagnostic_timing_is_logged_but_excluded_from_reported_runtime(
    tags: list[str], tmp_path: Path
) -> None:
    trials = [
        TrialResult(
            config={"x": i},
            config_id=str(i),
            config_tag=tag,
            val_score=0.7,
            test_score=0.8,
            fit_time_tuning=10.0,
            predict_time_tuning=2.0,
            fit_time_refit=3.0,
            predict_time_refit=0.5,
        )
        for i, tag in enumerate(tags)
    ]
    summary = ExperimentSummary(
        model_name="model",
        dataset="dataset",
        task_name="task",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_name="roc_auc",
        seed=0,
        n_trials=10,
        default=trials[0] if tags[0] == "default" else None,
        tuned=trials[-1],
        trials=trials,
    )
    path = tmp_path / "results.csv"
    summary_to_dataframe(summary).to_csv(path, index=False)
    results = pd.read_csv(path)
    assert results["val_score"].eq(0.7).all()
    assert results["fit_time_tuning"].eq(10.0).all()
    assert results["predict_time_tuning"].eq(2.0).all()
    row = to_bencheval_frame(results).iloc[0]
    assert row.time_train_s == (3.0 if tags == ["default"] else 13.0)
    assert row.time_infer_s == 0.5
