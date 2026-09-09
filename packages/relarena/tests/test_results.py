"""Benchmark result tables."""

import numpy as np

from relarena.results import trials_to_dataframe
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
