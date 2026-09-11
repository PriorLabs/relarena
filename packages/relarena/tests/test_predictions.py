"""Prediction artifact round trips through the real tuner and evaluator."""

from __future__ import annotations

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
    InnerSplit,
    OuterSplit,
    RelArenaModel,
    RunIdentity,
    load_prediction_labels,
    load_predictions,
    prediction_context,
    runner,
)
from relarena.models.dummy import DummyBaseline
from relarena.search_space import SearchSpace


def _table(targets: list[float], date: str) -> Table:
    return Table(
        df=pd.DataFrame(
            {"entity": range(len(targets)), "time": pd.Timestamp(date), "y": targets}
        ),
        fkey_col_to_pkey_table={"entity": "entities"},
        pkey_col=None,
        time_col="time",
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
    source = SimpleNamespace(
        task=task,
        dataset_name="small",
        task_name="target",
        metric=roc_auc if binary else mae,
        inner_split=lambda: inner,
        outer_split=lambda: outer,
        run_identity=identity.for_phase,
    )
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
        root = tmp_path / Constant.name / "small" / "target" / "0" / trial.config_id
        for split, split_object, target in [("val", inner, val), ("test", outer, test)]:
            path = root / f"{split}.npz"
            additional = trial.config["constant"] == 2
            if split == "test" and additional and not all_configs:
                assert not path.exists()
                continue
            context = prediction_context(
                task, split_object, identity.for_phase(split_object.name)
            )
            predictions, metadata = load_predictions(path, expected_context=context)
            labels = load_prediction_labels(path)
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
                    load_predictions(path, expected_context=mismatched)
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
            / DummyBaseline.name
            / "small"
            / "target"
            / "0"
            / builtin.default.config_id
            / f"{split}.npz"
        )
        predictions, metadata = load_predictions(
            path,
            expected_context=prediction_context(
                task, split_object, identity.for_phase(split_object.name)
            ),
        )
        assert task.evaluate(predictions, target) == pytest.approx(metadata["metrics"])

    assert len(list((tmp_path / "labels").glob("*.npz"))) == 2
    label_file = path.parent / metadata["labels_path"]
    with np.load(label_file, allow_pickle=False) as artifact:
        labels = artifact["labels"].copy()
        context_array = artifact["context"]
    labels[0] += 1
    np.savez_compressed(label_file, labels=labels, context=context_array)
    with pytest.raises(ValueError, match="content or row identities"):
        load_prediction_labels(path)
