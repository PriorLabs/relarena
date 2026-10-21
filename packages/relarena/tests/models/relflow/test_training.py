"""Exercise complete validation selection and actual tiny CPU checkpoint reloads."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

rf = pytest.importorskip("relflow")

from relarena.models.relflow import training  # noqa: E402


def recipe(kind: str = "regression") -> dict:
    return {
        "contract": {"task_type": kind},
        "model": {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "batch_size": 4,
            "fields": {
                "value": {"type": "number"},
                "target": {
                    "type": "boolean" if kind == "binary_classification" else "number",
                    "mask": True,
                },
            },
        },
        "training": {
            "min_steps": 2,
            "min_full_passes": 0,
            "patience": 2,
            "min_delta": 0.0,
            "max_epochs": 1,
            "max_steps": 2,
            "gradient_clip_val": 1.0,
            "val_every": 1,
        },
        "optimizer": {"learning_rate": 1e-3, "weight_decay": 0.01, "fused": False},
    }


def write_rows(path: Path, labels: list, *, masked: bool = False) -> Path:
    data = {
        "__row_position": list(range(len(labels))),
        "value": [float(i) for i in range(len(labels))],
    }
    if not masked:
        data["target"] = labels
    pq.write_table(pa.table(data), path)
    return path


@pytest.mark.parametrize("kind", ["regression", "binary_classification"])
def test_tiny_cpu_fit_reload_and_masked_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    labels = [0.0, 1.0] * 4 if kind == "regression" else [False, True] * 4
    train_path = write_rows(tmp_path / "train.parquet", labels)
    validation_path = write_rows(tmp_path / "validation.parquet", labels[:6])
    masked_path = write_rows(tmp_path / "masked.parquet", labels[:3], masked=True)
    result = training.train(
        train_path, validation_path, recipe(kind), tmp_path / "fit", seed=0
    )
    assert (tmp_path / "fit" / "best.ckpt").is_file()
    predictions = training.predict(result, masked_path, seed=0)
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    if kind == "binary_classification":
        assert ((predictions >= 0) & (predictions <= 1)).all()
    reloaded = rf.Model.load(tmp_path / "fit" / "best.ckpt")
    np.testing.assert_allclose(
        training.predict(reloaded, masked_path, seed=0), predictions
    )


def test_inference_stream_never_exposes_target(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "rows.parquet", [0, 1, 0])
    batch = next(training.batches(path, predict=True))
    assert batch.schema.names == ["__row_position", "value"]


def test_metric_is_full_split_auc_and_source_unit_mae() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    assert training.score("binary_classification", labels, predictions) == 0.75
    assert training.score(
        "regression", np.array([0.0, 10.0, 100.0]), np.zeros(3)
    ) == pytest.approx(110 / 3)


@pytest.mark.parametrize(
    ("kind", "labels", "values"),
    [
        ("regression", [], []),
        ("regression", [0, 1], [0]),
        ("regression", [float("nan")], [0]),
        ("regression", [1], [float("inf")]),
        ("binary_classification", [0, 0], [0.1, 0.2]),
        ("binary_classification", [0, 1], [-0.1, 0.5]),
    ],
)
def test_metric_rejects_invalid_or_incomplete_vectors(
    kind: str, labels: list[float], values: list[float]
) -> None:
    with pytest.raises(ValueError):
        training.score(kind, np.array(labels), np.array(values))


def test_selection_scores_all_rows_before_checkpointing(tmp_path: Path) -> None:
    selected = training.Selection(
        recipe("binary_classification"), 8, 4, tmp_path / "best.ckpt", math.inf
    )
    saves = []
    model = SimpleNamespace(save=lambda path: saves.append(path))
    trainer = SimpleNamespace(global_step=1, should_stop=False)
    selected.positions = [np.array([0, 1]), np.array([2, 3])]
    selected.labels = [np.array([0, 1]), np.array([0, 1])]
    selected.values = [np.array([0.1, 0.2]), np.array([0.8, 0.9])]
    selected.on_validation_epoch_end(trainer, model)
    assert selected.best == 0.75  # Both individual batches have AUROC 1.0.
    assert saves == [tmp_path / "best.ckpt"]


def test_patience_cannot_stop_before_minimum_training_passes(tmp_path: Path) -> None:
    settings = recipe()
    settings["training"].update(min_steps=1, min_full_passes=2, patience=1)
    selected = training.Selection(settings, 9, 2, tmp_path / "best.ckpt", math.inf)
    assert selected.minimum_steps == 6  # ceil(9/4) batches times two passes.
    model = SimpleNamespace(save=lambda path: None)
    trainer = SimpleNamespace(global_step=1, should_stop=False)
    for step in (1, 2, 5):
        trainer.global_step = step
        selected.positions = [np.array([0, 1])]
        selected.labels = [np.array([0.0, 1.0])]
        selected.values = [np.array([0.0, 0.0])]
        selected.on_validation_epoch_end(trainer, model)
        assert not trainer.should_stop
    trainer.global_step = 6
    selected.on_validation_epoch_end(trainer, model)
    assert trainer.should_stop


@pytest.mark.parametrize("positions", [[0, 0], [0], [0, 2]])
def test_selection_rejects_duplicated_missing_or_wrong_rows(
    tmp_path: Path, positions: list[int]
) -> None:
    selected = training.Selection(recipe(), 4, 2, tmp_path / "best.ckpt", math.inf)
    selected.positions = [np.array(positions)]
    selected.labels = [np.zeros(len(positions))]
    selected.values = [np.zeros(len(positions))]
    with pytest.raises(RuntimeError, match="every validation row once"):
        selected.on_validation_epoch_end(SimpleNamespace(global_step=1), object())


@pytest.mark.parametrize(
    ("positions", "values"),
    [
        ([0, 0, 2], [0.1, 0.2, 0.3]),
        ([0, 1], [0.1, 0.2]),
        ([0, 1, 2], [0.1, float("nan"), 0.3]),
    ],
)
def test_predict_rejects_invalid_output_axis_or_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    positions: list[int],
    values: list[float],
) -> None:
    path = write_rows(tmp_path / "rows.parquet", [0, 1, 0], masked=True)
    output = pa.table(
        {
            "inputs": [{"__row_position": position} for position in positions],
            "predictions": [{"target": {"content": value}} for value in values],
        }
    )
    model = SimpleNamespace(predict=lambda **kwargs: output)
    monkeypatch.setattr(training, "reset_text_cache", lambda model: None)
    with pytest.raises(RuntimeError, match="cover each anchor once"):
        training.predict(model, path, seed=0)


def test_predict_restores_anchor_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_rows(tmp_path / "rows.parquet", [0, 1, 0], masked=True)
    output = pa.table(
        {
            "inputs": [
                {"__row_position": 2},
                {"__row_position": 0},
                {"__row_position": 1},
            ],
            "predictions": [
                {"target": {"content": 0.7}},
                {"target": {"content": 0.2}},
                {"target": {"content": 0.4}},
            ],
        }
    )
    model = SimpleNamespace(predict=lambda **kwargs: output)
    monkeypatch.setattr(training, "reset_text_cache", lambda model: None)
    np.testing.assert_allclose(training.predict(model, path, seed=0), [0.2, 0.4, 0.7])
