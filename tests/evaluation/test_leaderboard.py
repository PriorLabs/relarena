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


#: The task type each primary metric belongs to (`roc_auc` is the binary
#: primary, `mae` the regression one), so `_result` rows carry a `task_type`
#: the subset filters can read.
_TASK_TYPE_OF = {"roc_auc": "BINARY_CLASSIFICATION", "mae": "REGRESSION"}


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
        "task_type": _TASK_TYPE_OF[metric],
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


def test__to_bencheval_frame__system_total_time__is_preserved() -> None:
    results = pd.DataFrame(
        [
            {
                "model": "a-system",
                "dataset": "rel-f1",
                "task": "driver-dnf",
                "metric": "roc_auc",
                "selected": True,
                "test_score": 0.9,
                "time_total": 12.5,
            }
        ]
    )

    row = to_bencheval_frame(results).iloc[0]

    assert row["time_train_s"] == pytest.approx(12.5)
    assert row["time_infer_s"] == pytest.approx(0.0)


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


def test__compute_leaderboard__incomplete_method__is_dropped() -> None:
    pytest.importorskip("bencheval.evaluator")
    # lightgbm is missing rel-hm/user-churn -> lightgbm goes, both tasks stay,
    # so one method's partial sweep never shrinks the board for the others.
    results = pd.DataFrame(
        [
            _result("constant-global", "rel-f1", "driver-dnf", "roc_auc", 0.5),
            _result("lightgbm", "rel-f1", "driver-dnf", "roc_auc", 0.9),
            _result("constant-global", "rel-hm", "user-churn", "mae", 0.8),
        ]
    )

    board = compute_leaderboard(results)
    assert set(board.index) == {"constant-global"}


def _mixed_results() -> pd.DataFrame:
    """Two task types x two methods, plus one method that only does regression."""
    rows = []
    for model, score in (("constant-global", 0.5), ("lightgbm", 0.9)):
        rows.append(_result(model, "rel-f1", "driver-dnf", "roc_auc", score))
        rows.append(_result(model, "rel-hm", "user-churn", "mae", 1.0 - score))
    rows.append(_result("regression-only", "rel-hm", "user-churn", "mae", 0.05))
    return pd.DataFrame(rows)


def test__compute_leaderboard__subset__ranks_only_that_task_type() -> None:
    pytest.importorskip("bencheval.evaluator")
    results = _mixed_results()

    classification = compute_leaderboard(results, subset="classification")

    # the regression-only method never entered, and neither did its task
    assert set(classification.index) == {"constant-global", "lightgbm"}
    assert classification.loc["lightgbm", "rank"] == pytest.approx(1.0)


def test__compute_leaderboard__narrow_method__ranks_only_on_its_subset() -> None:
    pytest.importorskip("bencheval.evaluator")
    # A method covering one task type only: absent from the full board (it
    # cannot cover it), ranked on the subset it does cover, and the full board
    # is the same with or without it.
    results = _mixed_results()
    without = results[results["model"] != "regression-only"]

    full = compute_leaderboard(results)
    regression = compute_leaderboard(results, subset="regression")

    assert "regression-only" not in set(full.index)
    assert set(full.index) == set(compute_leaderboard(without).index)
    assert set(regression.index) == {
        "constant-global",
        "lightgbm",
        "regression-only",
    }
    # it wins that board: mae 0.05 beats both
    assert regression.loc["regression-only", "rank"] == pytest.approx(1.0)


def test__compute_leaderboard__unknown_subset__raises() -> None:
    pytest.importorskip("bencheval.evaluator")
    with pytest.raises(ValueError, match="Unknown task subset 'tiny'"):
        compute_leaderboard(_mixed_results(), subset="tiny")


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
