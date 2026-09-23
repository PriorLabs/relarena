"""Prediction artifact round trips through the real tuner and evaluator."""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, EntityTask, Table, TaskType
from relbench.metrics import mae, roc_auc

from relarena import (
    load_prediction_labels,
    load_predictions,
    prediction_context,
    runner,
)
from relarena.models.dummy import DummyBaseline
from relarena.predictions import PredictionArtifactWriter
from relarena_core import InnerSplit, OuterSplit, RelArenaModel, RunIdentity
from relarena_core.search_space import SearchSpace


def _table(targets: list[float], date: str) -> Table:
    return Table(
        df=pd.DataFrame(
            {"entity": range(len(targets)), "time": pd.Timestamp(date), "y": targets}
        ),
        fkey_col_to_pkey_table={"entity": "entities"},
        pkey_col=None,
        time_col="time",
    )


def _source(binary: bool) -> SimpleNamespace:
    train = _table([0.0, 1.0] if binary else [0.0, 2.0], "2020-01-01")
    val = _table([0.0, 1.0] if binary else [1.0, 1.0], "2020-02-01")
    test = _table([0.0, 1.0] if binary else [0.5, 1.5], "2020-03-01")
    masked = Table(
        df=test.df.drop(columns="y"),
        fkey_col_to_pkey_table={"entity": "entities"},
        pkey_col=None,
        time_col="time",
    )
    task = SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION if binary else TaskType.REGRESSION,
        entity_col="entity",
        time_col="time",
        target_col="y",
        metrics=[roc_auc] if binary else [mae],
    )
    task.get_table = lambda *a, **k: test
    task.evaluate = lambda *a, **k: EntityTask.evaluate(task, *a, **k)
    db = Database({})
    inner = InnerSplit(db, pd.Timestamp("2020-02-01"), train, val, val)
    outer = OuterSplit(db, pd.Timestamp("2020-03-01"), train, masked, val)
    identity = RunIdentity("small", "db", "target", "task")
    return SimpleNamespace(
        task=task,
        dataset_name="small",
        task_name="target",
        metric=roc_auc if binary else mae,
        inner_split=lambda: inner,
        outer_split=lambda: outer,
        run_identity=identity.for_phase,
    )


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("all_configs", [False, True])
@pytest.mark.parametrize("refit_full", [False, True])
def test_artifact_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    all_configs: bool,
    refit_full: bool,
    binary: bool,
) -> None:
    source = _source(binary)
    task = source.task
    inner, outer = source.inner_split(), source.outer_split()
    val, test = inner.eval_target, task.get_table("test")
    identity = source.run_identity("inner")
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: source)
    fits: list[tuple[int, bool]] = []

    class Constant(RelArenaModel):
        name = "artifact-constant"
        refit_on_full_data = refit_full

        def fit(
            self,
            task: Any,
            db: Any,
            train_table: Table,
            val_table: Table | None,
            **kwargs: Any,
        ) -> None:
            assert (train_table.df["time"] < outer.cutoff).all()
            fits.append((len(train_table.df), val_table is None))

        def predict(self, task: Any, db: Any, table: Table) -> np.ndarray:
            if table.df["time"].iloc[0] == outer.cutoff:
                assert "y" not in table.df
            if binary:
                return np.array([[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]])[
                    self.config["constant"]
                ]
            return np.full(len(table.df), self.config["constant"], dtype=float)

    summary = runner.run_experiment(
        Constant,
        "small",
        "target",
        search_space=SearchSpace(
            fixed_grid=[{"constant": 0}, {"constant": 1}, {"constant": 2}],
            default_overrides={"constant": 0},
        ),
        n_trials=3,
        predictions_dir=tmp_path,
        refit_all_configs=all_configs,
    )
    assert len(fits) == (6 if all_configs else 5)
    assert fits[:3] == [(2, False)] * 3
    assert fits[3:] == ([(4, True)] if refit_full else [(2, False)]) * (len(fits) - 3)
    for trial in summary.trials:
        root = (
            tmp_path
            / "data"
            / f"{Constant.name}_{trial.config_id}"
            / "small__target"
            / "0_0"
        )
        for split, split_object, target in [("val", inner, val), ("test", outer, test)]:
            additional = trial.config["constant"] == 2
            complete = all_configs or not additional
            path = root / ("results.pkl" if complete else "validation.partial")
            if split == "test" and not complete:
                assert not (root / "results.pkl").exists()
                continue
            context = prediction_context(
                task, split_object, identity.for_phase(split_object.name)
            )
            predictions, metadata = load_predictions(
                path, split=split, expected_context=context
            )
            labels = load_prediction_labels(path, split=split)
            np.testing.assert_array_equal(labels, target.df["y"].to_numpy())
            assert {
                m.__name__: m(labels, predictions) for m in task.metrics
            } == pytest.approx(metadata["metrics"])
            assert metadata["additional_refit"] == (additional and split == "test")
            assert metadata["fit_time"] >= 0
            assert metadata["predict_time"] >= 0
            for field, value in [
                ("rows", list(reversed(context["rows"]))),
                ("split", "wrong"),
                ("classes", [1, 0]),
            ]:
                mismatched = deepcopy(context)
                mismatched[field] = value
                with pytest.raises(ValueError, match="differ"):
                    load_predictions(path, split=split, expected_context=mismatched)
        if trial.config["constant"] == 2:
            assert trial.test_pred is None
            assert trial.test_score is None
            assert trial.fit_time_refit is None

    builtin = runner.run_experiment(
        DummyBaseline,
        "small",
        "target",
        predictions_dir=tmp_path,
    )
    for split, split_object, target in [("val", inner, val), ("test", outer, test)]:
        path = (
            tmp_path
            / "data"
            / f"{DummyBaseline.name}_{builtin.default.config_id}"
            / "small__target"
            / "0_0"
            / "results.pkl"
        )
        predictions, metadata = load_predictions(
            path,
            split=split,
            expected_context=prediction_context(
                task, split_object, identity.for_phase(split_object.name)
            ),
        )
        assert task.evaluate(predictions, target) == pytest.approx(metadata["metrics"])

    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    assert artifact["framework"] == f"{DummyBaseline.name}_{builtin.default.config_id}"
    assert artifact["problem_type"] == ("binary" if binary else "regression")
    assert artifact["task_metadata"] == {
        "name": "small__target",
        "fold": 0,
        "repeat": 0,
        "sample": 0,
        "split_idx": 0,
    }
    assert artifact["relarena"]["validation_protocol"] == "temporal_holdout"
    assert artifact["metric_error"] == pytest.approx(
        1 - builtin.default.test_score if binary else builtin.default.test_score
    )
    assert artifact["metric_error_val"] == pytest.approx(
        1 - builtin.default.val_score if binary else builtin.default.val_score
    )
    assert artifact["time_train_s"] == pytest.approx(
        builtin.default.fit_time_tuning + builtin.default.fit_time_refit
    )
    assert artifact["time_infer_s"] == builtin.default.predict_time_refit
    assert (
        artifact["method_metadata"]["model_hyperparameters"] == builtin.default.config
    )
    for split in ("val", "test"):
        sim = artifact["simulation_artifacts"]
        np.testing.assert_array_equal(sim[f"y_{split}_idx"], np.arange(2))
        assert sim[f"pred_{split}"].shape == sim[f"y_{split}"].shape == (2,)
    assert not list(path.parent.glob("validation.partial"))
    artifact["simulation_artifacts"]["y_test"][0] += 1
    with path.open("wb") as stream:
        pickle.dump(artifact, stream)
    with pytest.raises(ValueError, match="checksum"):
        load_prediction_labels(path, split="test")


@pytest.mark.parametrize(
    "task_type",
    [
        task_type
        for task_type in TaskType
        if task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION)
    ],
)
def test_artifacts_reject_unsupported_task_types(
    tmp_path: Path, task_type: TaskType
) -> None:
    table = _table([0.0, 1.0], "2020-01-01")
    split = InnerSplit(Database({}), pd.Timestamp("2020-02-01"), table, table, table)
    source = SimpleNamespace(task=SimpleNamespace(task_type=task_type), metric=mae)
    writer = PredictionArtifactWriter(tmp_path, source, "test-model", 0, False)
    with pytest.raises(
        ValueError, match="Prediction artifacts do not support task type"
    ):
        writer.save_validation([], split)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["refit", "save"])
def test_extra_artifact_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    source = _source(False)
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: source)
    original_refit = runner.refit_and_evaluate
    original_save = PredictionArtifactWriter.save_test
    refits = []

    def refit(model: Any, config: dict, **kwargs: Any) -> dict:
        refits.append(config["candidate"])
        if failure == "refit" and config["candidate"] == 1:
            raise RuntimeError("extra refit failed")
        return original_refit(model, config, **kwargs)

    def save(
        writer: PredictionArtifactWriter, trial: Any, *args: Any, **kwargs: Any
    ) -> None:
        if failure == "save" and trial.config["candidate"] == 1:
            raise OSError("extra export failed")
        original_save(writer, trial, *args, **kwargs)

    monkeypatch.setattr(runner, "refit_and_evaluate", refit)
    monkeypatch.setattr(PredictionArtifactWriter, "save_test", save)
    grid = [{"candidate": i} for i in (0, 1, 2)]
    error = RuntimeError if failure == "refit" else OSError
    with pytest.raises(error, match="extra .* failed"):
        runner.run_model_experiment(
            DummyBaseline,
            "small",
            "target",
            predictions_dir=tmp_path,
            search_space=SearchSpace(fixed_grid=grid, default_overrides=grid[0]),
            n_trials=3,
            refit_all_configs=True,
        )
    assert refits == [0, 1]
    assert len(list(tmp_path.rglob("results.pkl"))) == 1
    assert len(list(tmp_path.rglob("validation.partial"))) == 2


def test_validation_export_replaces_partial_but_preserves_completed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(False)
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: source)
    for _ in range(2):
        runner.run_model_experiment(
            DummyBaseline,
            "small",
            "target",
            predictions_dir=tmp_path,
            evaluate_test=False,
        )
    partial = next(tmp_path.rglob("validation.partial"))
    partial.write_bytes(b"interrupted write")
    runner.run_model_experiment(
        DummyBaseline,
        "small",
        "target",
        predictions_dir=tmp_path,
    )
    assert not partial.exists()
    complete = next(tmp_path.rglob("results.pkl"))
    with complete.open("rb") as stream:
        artifact = pickle.load(stream)
    assert "pred_test" in artifact["simulation_artifacts"]
    before = complete.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_model_experiment(
            DummyBaseline,
            "small",
            "target",
            predictions_dir=tmp_path,
        )
    assert complete.read_bytes() == before


def test_failed_partial_write_preserves_previous_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(False)
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *a, **k: source)
    runner.run_model_experiment(
        DummyBaseline,
        "small",
        "target",
        predictions_dir=tmp_path,
        evaluate_test=False,
    )
    partial = next(tmp_path.rglob("validation.partial"))
    before = partial.read_bytes()

    def fail_dump(obj: Any, stream: Any, **kwargs: Any) -> None:
        stream.write(b"incomplete pickle")
        raise OSError("disk full")

    monkeypatch.setattr(pickle, "dump", fail_dump)
    with pytest.raises(OSError, match="disk full"):
        runner.run_model_experiment(
            DummyBaseline,
            "small",
            "target",
            predictions_dir=tmp_path,
            evaluate_test=False,
        )
    assert partial.read_bytes() == before
    assert list(partial.parent.iterdir()) == [partial]
