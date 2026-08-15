"""Tests for relarena.evaluation.plots (smoke: figures are produced, files written).

The plotting stack is the `plots` extra; skip cleanly when it's absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("bencheval.evaluator")  # heatmaps use bencheval loss_rescaled

from relarena.evaluation import (
    plot_normalized_loss_heatmap,
    write_leaderboard_plots,
)


def _result(
    model: str, dataset: str, task: str, task_type: str, metric: str, test_score: float
) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "task": task,
        "task_type": task_type,
        "metric": metric,
        "selected": True,
        "test_score": test_score,
        "fit_time_tuning": 1.0,
        "fit_time_refit": 0.5,
        "predict_time_refit": 0.2,
    }


def _dense_results() -> pd.DataFrame:
    rows = []
    for i, model in enumerate(["constant-global", "lightgbm", "rdblearn"]):
        for task in ("a", "b", "c"):
            rows.append(
                _result(
                    model,
                    "rel-d",
                    task,
                    "BINARY_CLASSIFICATION",
                    "roc_auc",
                    0.5 + 0.1 * i,
                )
            )
        for task in ("x", "y", "z"):
            rows.append(
                _result(model, "rel-d", task, "REGRESSION", "mae", 1.0 - 0.2 * i)
            )
    df = pd.DataFrame(rows)
    # native-metric columns the raw heatmaps read (NaN where the metric doesn't apply)
    is_cls = df["task_type"] == "BINARY_CLASSIFICATION"
    df["test_roc_auc"] = df["test_score"].where(is_cls)
    df["test_r2"] = (1.0 - df["test_score"]).where(~is_cls)
    return df


def test__plot_normalized_loss_heatmap__no_tasks_of_type__writes_nothing(
    tmp_path: Path,
) -> None:
    regression_only = _dense_results().query("task_type == 'REGRESSION'")
    out = tmp_path / "heatmap.png"
    assert (
        plot_normalized_loss_heatmap(regression_only, "BINARY_CLASSIFICATION", out)
        is False
    )
    assert not out.exists()


def test__write_leaderboard_plots__dense_results__writes_heatmaps_and_cd(
    tmp_path: Path,
) -> None:
    pytest.importorskip("bencheval.evaluator")
    pytest.importorskip("autorank")

    written = write_leaderboard_plots(_dense_results(), tmp_path)

    names = {p.name for p in written}
    assert names == {
        "heatmap_classification.png",
        "heatmap_regression.png",
        "heatmap_roc_auc.png",
        "heatmap_r2.png",
        "winrate_matrix.png",
        "critical_difference.png",
    }
    for path in written:
        assert path.exists() and path.stat().st_size > 0
