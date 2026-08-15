"""Tests for the RelArena → bencheval results adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from relarena.evaluation import (
    compute_leaderboard,
    load_reference_results,
    to_bencheval_frame,
)

_BASELINE_DIR = Path(__file__).resolve().parents[2] / "baseline_results"
_REFERENCE_CSV = _BASELINE_DIR / "reference_results.csv"


def _result(
    model: str,
    dataset: str,
    task: str,
    metric: str,
    test_score: float,
    *,
    selected: bool = True,
) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "task": task,
        "metric": metric,
        "selected": selected,
        "test_score": test_score,
        "fit_time_tuning": 1.0,
        "fit_time_refit": 0.5,
        "predict_time_refit": 0.2,
    }


def test__to_bencheval_frame__mixed_runs__maps_columns_and_drops_unscored() -> None:
    results = pd.DataFrame(
        [
            _result("constant-global", "rel-f1", "driver-dnf", "roc_auc", 0.9),
            # a non-selected config for the same run is ignored
            _result(
                "constant-global",
                "rel-f1",
                "driver-dnf",
                "roc_auc",
                float("nan"),
                selected=False,
            ),
            _result("lightgbm", "rel-hm", "user-churn", "mae", 0.5),
            # a selected-but-unscored (failed/test-skipped) run is dropped
            _result("rdblearn", "rel-hm", "user-churn", "mae", float("nan")),
        ]
    )

    frame = to_bencheval_frame(results)

    assert list(frame.columns) == [
        "method",
        "task",
        "metric_error",
        "time_train_s",
        "time_infer_s",
    ]
    assert len(frame) == 2  # the non-selected and the unscored rows are gone
    assert set(frame["task"]) == {"rel-f1/driver-dnf", "rel-hm/user-churn"}

    constant = frame[frame["method"] == "constant-global"].iloc[0]
    assert constant["task"] == "rel-f1/driver-dnf"
    assert constant["metric_error"] == pytest.approx(0.1)  # roc_auc: 1 - 0.9
    assert constant["time_train_s"] == pytest.approx(1.5)  # tuning 1.0 + refit 0.5
    assert constant["time_infer_s"] == pytest.approx(0.2)

    lgbm = frame[frame["method"] == "lightgbm"].iloc[0]
    assert lgbm["metric_error"] == pytest.approx(0.5)  # mae is its own error
    assert (frame["metric_error"] >= 0).all()


def test__to_bencheval_frame__dense_matrix__passes_bencheval_verify_data() -> None:
    bencheval = pytest.importorskip("bencheval.evaluator")

    # Dense 2-method × 2-task matrix.
    results = pd.DataFrame(
        [
            _result("constant-global", "rel-f1", "driver-dnf", "roc_auc", 0.7),
            _result("lightgbm", "rel-f1", "driver-dnf", "roc_auc", 0.9),
            _result("constant-global", "rel-hm", "user-churn", "mae", 0.8),
            _result("lightgbm", "rel-hm", "user-churn", "mae", 0.4),
        ]
    )

    frame = to_bencheval_frame(results)

    board = bencheval.BenchmarkEvaluator()  # default method/task/metric_error cols
    board.verify_data(frame)  # raises if the contract is violated
    out = board.leaderboard(frame, include_elo=False, include_winrate=False)
    assert len(out) == 2  # one ranked row per method


def test__compute_leaderboard__dense_results__ranks_by_normalized_loss() -> None:
    pytest.importorskip("bencheval.evaluator")
    # constant-global is worst on both tasks, lightgbm best -> lightgbm ranks first.
    results = pd.DataFrame(
        [
            _result("constant-global", "rel-f1", "driver-dnf", "roc_auc", 0.5),
            _result("lightgbm", "rel-f1", "driver-dnf", "roc_auc", 0.95),
            _result("constant-global", "rel-hm", "user-churn", "mae", 1.0),
            _result("lightgbm", "rel-hm", "user-churn", "mae", 0.2),
        ]
    )

    board = compute_leaderboard(results)

    assert list(board.index) == [
        "lightgbm",
        "constant-global",
    ]  # sorted by loss_rescaled
    assert {"loss_rescaled", "rank", "winrate", "elo"}.issubset(board.columns)
    assert board.loc["constant-global", "elo"] == pytest.approx(
        1000.0
    )  # anchored on constant-global


def test__compute_leaderboard__single_method__ranks_without_elo() -> None:
    pytest.importorskip("bencheval.evaluator")
    # A one-model sweep: Elo has no pair to compare, but the board still ranks.
    results = pd.DataFrame(
        [
            _result("lightgbm", "rel-f1", "driver-dnf", "roc_auc", 0.95),
            _result("lightgbm", "rel-hm", "user-churn", "mae", 0.2),
        ]
    )

    board = compute_leaderboard(results)

    assert list(board.index) == ["lightgbm"]
    assert "elo" not in board.columns
    assert board.loc["lightgbm", "rank"] == pytest.approx(1.0)


def test__compute_leaderboard__incomplete_task__is_dropped() -> None:
    pytest.importorskip("bencheval.evaluator")
    # rel-hm/user-churn is missing lightgbm -> dropped, leaving one dense task.
    results = pd.DataFrame(
        [
            _result("constant-global", "rel-f1", "driver-dnf", "roc_auc", 0.5),
            _result("lightgbm", "rel-f1", "driver-dnf", "roc_auc", 0.9),
            _result("constant-global", "rel-hm", "user-churn", "mae", 0.8),
        ]
    )

    board = compute_leaderboard(results)
    assert set(board.index) == {
        "constant-global",
        "lightgbm",
    }  # both methods still ranked


def test__compute_leaderboard__checked_in_baselines__produce_a_board() -> None:
    # Guard: the committed baseline snapshot must always rank into a valid,
    # dense leaderboard. Catches format drift (an unregistered metric, a method
    # missing tasks) before it reaches the board.
    pytest.importorskip("bencheval.evaluator")
    path = _BASELINE_DIR / "results.csv"
    if not path.exists():
        pytest.skip("no committed baseline_results/results.csv")

    board = compute_leaderboard(pd.read_csv(path))
    assert not board.empty
    assert "loss_rescaled" in board.columns


def test__compute_leaderboard__with_reference__ranks_reported_as_baselines() -> None:
    pytest.importorskip("bencheval.evaluator")
    path = _BASELINE_DIR / "results.csv"
    if not path.exists():
        pytest.skip("no committed baseline_results/results.csv")

    reference = load_reference_results(
        _REFERENCE_CSV, methods=["graphsage_MR", "rdblearn_MR"]
    )
    board = compute_leaderboard(pd.read_csv(path), reference=reference)
    # reference methods rank alongside the reproduced models, none dropped
    assert {"graphsage_MR", "rdblearn_MR"}.issubset(board.index)
    assert {"graphsage", "rdblearn"}.issubset(board.index)


def test__method_kind__cold_process__reports_the_registered_kind() -> None:
    # `method_kind` must register the built-ins itself: a leaderboard-only
    # caller (load a results CSV, rank it) never imports `relarena.models`, and
    # with an empty registry every lookup would fall through to "model" — a
    # system would silently join the models-only board. A subprocess is the
    # only honest cold path; in-process, other tests have already registered.
    import subprocess
    import sys

    probe = (
        "from relarena.evaluation import method_kind; "
        "assert method_kind('rt-plurel') == 'system', method_kind('rt-plurel'); "
        "assert method_kind('lightgbm') == 'model'; "
        "assert method_kind('not-registered') == 'model'"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test__compute_leaderboard__kinds_filter__excludes_systems() -> None:
    pytest.importorskip("bencheval.evaluator")
    results = pd.DataFrame(
        [
            _result("constant-global", "rel-f1", "driver-dnf", "roc_auc", 0.5),
            _result("lightgbm", "rel-f1", "driver-dnf", "roc_auc", 0.9),
            _result("rt-plurel", "rel-f1", "driver-dnf", "roc_auc", 0.95),
        ]
    )

    combined = compute_leaderboard(results)
    models_only = compute_leaderboard(results, kinds=frozenset({"model"}))

    assert "rt-plurel" in set(combined.index)
    assert set(models_only.index) == {"constant-global", "lightgbm"}
