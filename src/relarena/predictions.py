"""Portable per-configuration prediction artifacts for benchmark models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from relbench.base import EntityTask, Table, TaskType

from relarena.dataset import InnerSplit, OuterSplit, RelBenchDatasetTask, Split
from relarena.identity import RunIdentity
from relarena.metrics import get_metric
from relarena.results import TrialResult


def prediction_context(
    task: EntityTask, split: Split, identity: RunIdentity
) -> dict[str, Any]:
    """Describe ordered prediction rows and the temporal data view."""
    rows = split.eval_table.df[[task.entity_col, task.time_col]]
    return {
        "identity": asdict(identity),
        "split": split.name,
        "cutoff": split.cutoff.isoformat(),
        "row_columns": list(rows.columns),
        "rows": json.loads(
            rows.to_json(orient="values", date_format="iso", date_unit="ns")
        ),
        "task_type": task.task_type.name,
        "target": task.target_col,
        "classes": [0, 1] if task.task_type == TaskType.BINARY_CLASSIFICATION else None,
        "prediction_class": 1
        if task.task_type == TaskType.BINARY_CLASSIFICATION
        else None,
    }


def save_labels(
    directory: Path, task: EntityTask, target: Table, context: dict[str, Any]
) -> Path:
    """Store labels once by content identity, preserving prediction row order."""
    rows = json.loads(
        target.df[context["row_columns"]].to_json(
            orient="values", date_format="iso", date_unit="ns"
        )
    )
    if rows != context["rows"]:
        raise ValueError("Label rows differ from prediction rows.")
    labels = target.df[task.target_col].to_numpy()
    if labels.dtype.kind not in "bifu" or not np.isfinite(labels).all():
        raise ValueError("Labels must be finite numeric values.")
    encoded = json.dumps(context, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(
        encoded.encode() + labels.dtype.str.encode() + labels.tobytes()
    ).hexdigest()
    path = directory / f"{digest}.npz"
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            np.savez_compressed(stream, labels=labels, context=np.array(encoded))
    return path


def load_prediction_labels(path: Path) -> np.ndarray:
    """Load the shared labels referenced by a prediction artifact."""
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"]))
    label_path = path.parent / metadata["labels_path"]
    with np.load(label_path, allow_pickle=False) as artifact:
        labels = artifact["labels"]
        context = json.loads(str(artifact["context"]))
    encoded = json.dumps(context, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(
        encoded.encode() + labels.dtype.str.encode() + labels.tobytes()
    ).hexdigest()
    if context != metadata["context"] or digest != metadata["labels_id"]:
        raise ValueError("Label artifact content or row identities differ.")
    if labels.shape != (len(context["rows"]),):
        raise ValueError("Label shape does not match artifact rows.")
    return labels


def save_predictions(
    path: Path,
    predictions: np.ndarray,
    *,
    context: dict[str, Any],
    labels_path: Path,
    model: str,
    trial: TrialResult,
    seed: int,
    metric: str,
    metrics: dict[str, float],
    fit_regime: str,
    fit_time: float,
    predict_time: float,
    additional_refit: bool,
) -> None:
    """Write one prediction NPZ without overwriting an existing artifact."""
    predictions = np.asarray(predictions)
    if predictions.shape != (len(context["rows"]),):
        raise ValueError("Prediction shape does not match artifact rows.")
    if predictions.dtype.kind not in "fi" or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite numeric values.")
    if context["classes"] is not None and ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Binary predictions must be probabilities in [0, 1].")
    metadata = {
        "version": 1,
        "context": context,
        "labels_path": os.path.relpath(labels_path, path.parent),
        "labels_id": labels_path.stem,
        "model": model,
        "config": trial.config,
        "config_id": trial.config_id,
        "config_tag": trial.config_tag,
        "seed": seed,
        "metric": asdict(get_metric(metric)),
        "versions": {name: version(name) for name in ("relarena", "relbench", "numpy")},
        "metrics": metrics,
        "fit_regime": fit_regime,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "additional_refit": additional_refit,
        "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
    }
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, predictions=predictions, metadata=np.array(encoded))


def load_predictions(
    path: Path, *, expected_context: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load predictions only when ordered rows and split/class metadata match."""
    with np.load(path, allow_pickle=False) as artifact:
        predictions = artifact["predictions"]
        metadata = json.loads(str(artifact["metadata"]))
    if metadata["version"] != 1:
        raise ValueError("Unsupported prediction artifact version.")
    if metadata["context"] != expected_context:
        raise ValueError("Prediction artifact rows, split or class ordering differ.")
    if predictions.shape != (len(expected_context["rows"]),):
        raise ValueError("Prediction shape does not match artifact rows.")
    if (
        hashlib.sha256(predictions.tobytes()).hexdigest()
        != metadata["prediction_sha256"]
    ):
        raise ValueError("Prediction artifact checksum differs.")
    return predictions, metadata


@dataclass
class PredictionArtifactWriter:
    """Own the paths and shared labels for one model experiment."""

    directory: Path
    source: RelBenchDatasetTask
    model: str
    seed: int
    refit_on_full_data: bool
    _test_labels_path: Path | None = field(default=None, init=False)

    @property
    def root(self) -> Path:
        """Return the prediction directory for this model, task and seed."""
        return (
            self.directory
            / self.model
            / self.source.dataset_name
            / self.source.task_name
            / str(self.seed)
        )

    def save_validation(self, trials: list[TrialResult], split: InnerSplit) -> None:
        """Save tuning predictions with their shared validation labels."""
        context = prediction_context(
            self.source.task, split, self.source.run_identity("inner")
        )
        labels_path = save_labels(
            self.directory / "labels", self.source.task, split.eval_target, context
        )
        for trial in trials:
            if trial.ok and trial.val_pred is not None:
                save_predictions(
                    self.root / trial.config_id / "val.npz",
                    trial.val_pred,
                    context=context,
                    labels_path=labels_path,
                    model=self.model,
                    trial=trial,
                    seed=self.seed,
                    metric=self.source.metric.__name__,
                    metrics=trial.val_metrics,
                    fit_regime="train_with_validation",
                    fit_time=trial.fit_time_tuning,
                    predict_time=trial.predict_time_tuning,
                    additional_refit=False,
                )

    def save_test(
        self,
        trial: TrialResult,
        result: dict[str, Any],
        split: OuterSplit,
        *,
        additional: bool,
    ) -> None:
        """Save final-fit predictions and retrieve labels after prediction."""
        context = prediction_context(
            self.source.task, split, self.source.run_identity("outer")
        )
        if self._test_labels_path is None:
            self._test_labels_path = save_labels(
                self.directory / "labels",
                self.source.task,
                self.source.task.get_table("test", mask_input_cols=False),
                context,
            )
        save_predictions(
            self.root / trial.config_id / "test.npz",
            result["test_pred"],
            context=context,
            labels_path=self._test_labels_path,
            model=self.model,
            trial=trial,
            seed=self.seed,
            metric=self.source.metric.__name__,
            metrics=result["test_metrics"],
            fit_regime="train_plus_val"
            if self.refit_on_full_data
            else "train_with_validation",
            fit_time=result["fit_time_refit"],
            predict_time=result["predict_time_refit"],
            additional_refit=additional,
        )
