"""Tests for the reference-baseline loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from relarena.evaluation import load_reference_results, to_bencheval_frame

_BASELINE_DIR = Path(__file__).resolve().parents[2] / "baseline_results"
_REFERENCE_CSV = _BASELINE_DIR / "reference_results.csv"


def test__load_reference_results__all_methods__adds_adapter_columns() -> None:
    frame = load_reference_results(_REFERENCE_CSV)
    for col in (
        "model",
        "dataset",
        "task",
        "metric",
        "test_score",
        "selected",
        "fit_time_tuning",
        "fit_time_refit",
        "predict_time_refit",
    ):
        assert col in frame.columns
    assert frame["selected"].all()
    # Reported numbers have no timing; the adapter columns are NaN, not invented.
    assert frame["fit_time_tuning"].isna().all()


def test__load_reference_results__method_subset__filters_to_requested() -> None:
    frame = load_reference_results(_REFERENCE_CSV, methods=["graphsage_MR"])
    assert set(frame["model"]) == {"graphsage_MR"}


def test__load_reference_results__unknown_method__raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        load_reference_results(_REFERENCE_CSV, methods=["does_not_exist"])


def test__load_reference_results__feeds_bencheval_adapter__dense_and_valid() -> None:
    bencheval = pytest.importorskip("bencheval.evaluator")
    frame = load_reference_results(_REFERENCE_CSV)
    bench = to_bencheval_frame(frame)
    # Every reference method survives the adapter on every task (dense matrix).
    assert bench["method"].nunique() == frame["model"].nunique()
    assert (bench["metric_error"] >= 0).all()
    bencheval.BenchmarkEvaluator().verify_data(bench)


def test__load_reference_results__mirrors_score_into_native_metric_column() -> None:
    # Classification references carry test_roc_auc (so they show in the raw ROC AUC
    # heatmap); they have no R², so test_r2 is absent.
    frame = load_reference_results(_REFERENCE_CSV, methods=["graphsage_MR"])
    clf = frame[frame["metric"] == "roc_auc"]
    assert (clf["test_roc_auc"] == clf["test_score"]).all()
    assert "test_r2" not in frame.columns
