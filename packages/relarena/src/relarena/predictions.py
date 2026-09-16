"""Per-configuration prediction artifacts using the TabArena result schema.

PredictionArtifactWriter is the runner's entry point: save_validation writes
initial artifacts and save_test completes them with test predictions and labels.
For reading, load_predictions checks a split against prediction_context;
load_prediction_labels retrieves its labels. Both loaders require trusted files.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import numpy as np
from relbench.base import EntityTask, Table, TaskType

from relarena.dataset import InnerSplit, OuterSplit, RelBenchDatasetTask, Split
from relarena.identity import RunIdentity
from relarena.metrics import get_metric
from relarena.results import TrialResult


def prediction_context(
    task: EntityTask, split: Split, identity: RunIdentity
) -> dict[str, Any]:
    """Describe ordered prediction rows and the temporal data view.

    This RelArena-specific context lives outside TabArena's simulation fields.
    """
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


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(values.dtype.str.encode() + values.tobytes()).hexdigest()


def _save_result(path: Path, result: dict[str, Any]) -> None:
    """Serialize a result dictionary as a pickle for TabArena's result reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as stream:
        pickle.dump(result, stream, protocol=4)
        stream.flush()
        os.link(stream.name, path)


def load_predictions(
    path: Path,
    *,
    split: Literal["val", "test"],
    expected_context: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read trusted prediction artifacts.

    Load a split from a trusted pickle, checking row alignment and array integrity.
    Prediction arrays follow TabArena's simulation_artifacts schema; validation
    uses the additional relarena metadata, which this loader requires.
    """
    with path.open("rb") as stream:
        result = pickle.load(stream)
    if result["relarena"]["version"] != 1:
        raise ValueError("Unsupported prediction artifact version.")
    metadata = result["relarena"][split]
    if metadata["context"] != expected_context:
        raise ValueError("Prediction artifact rows, split or class ordering differ.")
    predictions = result["simulation_artifacts"][f"pred_{split}"]
    labels = result["simulation_artifacts"][f"y_{split}"]
    for values, key in [(predictions, "prediction_sha256"), (labels, "labels_sha256")]:
        if values.shape != (len(expected_context["rows"]),):
            raise ValueError("Array shape does not match artifact rows.")
        if _array_digest(values) != metadata[key]:
            raise ValueError("Prediction or label artifact checksum differs.")
    return predictions, metadata


def load_prediction_labels(path: Path, *, split: Literal["val", "test"]) -> np.ndarray:
    """Load labels from a trusted result pickle and verify their checksum."""
    with path.open("rb") as stream:
        result = pickle.load(stream)
    labels = result["simulation_artifacts"][f"y_{split}"]
    metadata = result["relarena"][split]
    if labels.shape != (len(metadata["context"]["rows"]),):
        raise ValueError("Label shape does not match artifact rows.")
    if _array_digest(labels) != metadata["labels_sha256"]:
        raise ValueError("Label artifact checksum differs.")
    return labels


@dataclass
class PredictionArtifactWriter:
    """Write self-contained config results with temporal provenance kept separately."""

    directory: Path
    source: RelBenchDatasetTask
    model: str
    seed: int
    refit_on_full_data: bool

    def _root(self, trial: TrialResult) -> Path:
        """Follow TabArena's data/method/task/repeat_fold directory layout.

        Each config is a method; the seed identifies a repeat of temporal fold 0.
        """
        return (
            self.directory
            / "data"
            / f"{self.model}_{trial.config_id}"
            / f"{self.source.dataset_name}__{self.source.task_name}"
            / f"{self.seed}_0"
        )

    def _add_split(
        self,
        result: dict[str, Any],
        split: Split,
        target: Table,
        predictions: np.ndarray,
        metrics: dict[str, float],
        fit_time: float,
        predict_time: float,
        additional: bool,
    ) -> None:
        """Use TabArena's pred_*, y_* and y_*_idx simulation field names.

        Indices are split-relative row positions. Entity IDs, row timestamps,
        and temporal split details are stored under relarena, alongside
        TabArena's standard prediction fields.
        """
        task = self.source.task
        context = prediction_context(task, split, self.source.run_identity(split.name))
        rows = json.loads(
            target.df[context["row_columns"]].to_json(
                orient="values", date_format="iso", date_unit="ns"
            )
        )
        if rows != context["rows"]:
            raise ValueError("Label rows differ from prediction rows.")
        predictions = np.asarray(predictions)
        labels = target.df[task.target_col].to_numpy()
        for values in (predictions, labels):
            if values.shape != (len(rows),):
                raise ValueError("Array shape does not match artifact rows.")
            if values.dtype.kind not in "bifu" or not np.isfinite(values).all():
                raise ValueError(
                    "Predictions and labels must be finite numeric values."
                )
        if (
            context["classes"] is not None
            and ((predictions < 0) | (predictions > 1)).any()
        ):
            raise ValueError("Binary predictions must be probabilities in [0, 1].")
        name = "val" if split.name == "inner" else "test"
        result["simulation_artifacts"].update(
            {
                f"pred_{name}": predictions,
                f"y_{name}": labels,
                f"y_{name}_idx": np.arange(len(rows)),
            }
        )
        result["relarena"][name] = {
            "context": context,
            "metrics": metrics,
            "fit_regime": "train_plus_val"
            if name == "test" and self.refit_on_full_data
            else "train_with_validation",
            "fit_time": fit_time,
            "predict_time": predict_time,
            "additional_refit": additional,
            "prediction_sha256": _array_digest(predictions),
            "labels_sha256": _array_digest(labels),
        }

    def save_validation(self, trials: list[TrialResult], split: InnerSplit) -> None:
        """Persist tuning outputs.

        Write validation-only artifacts until outer-test predictions are available.
        Metadata and class fields follow TabArena's config result schema. The
        .partial suffix keeps incomplete results out of pickle-file discovery.
        """
        task_type = self.source.task.task_type
        problem_types = {
            TaskType.BINARY_CLASSIFICATION: "binary",
            TaskType.REGRESSION: "regression",
        }
        if task_type not in problem_types:
            raise ValueError(
                f"Prediction artifacts do not support task type {task_type}."
            )
        problem_type = problem_types[task_type]
        binary = problem_type == "binary"
        metric = get_metric(self.source.metric)
        for trial in trials:
            if not trial.ok or trial.val_pred is None:
                continue
            result = {
                "framework": f"{self.model}_{trial.config_id}",
                "task_metadata": {
                    "name": f"{self.source.dataset_name}__{self.source.task_name}",
                    "fold": 0,
                    "repeat": self.seed,
                    "sample": 0,
                    "split_idx": self.seed,
                },
                "problem_type": problem_type,
                "metric": metric.name,
                "metric_error_val": metric.to_error(trial.val_score),
                "method_metadata": {
                    "model_type": self.model,
                    "model_hyperparameters": trial.config,
                    "name_prefix": self.model,
                },
                "simulation_artifacts": {
                    "label": self.source.task.target_col,
                    "num_classes": 2 if binary else None,
                    "ordered_class_labels": [0, 1] if binary else None,
                    "ordered_class_labels_transformed": [0, 1] if binary else None,
                },
                "relarena": {
                    "version": 1,
                    "validation_protocol": "temporal_holdout",
                    "seed": self.seed,
                    "config_id": trial.config_id,
                    "config_tag": trial.config_tag,
                    "metric": asdict(metric),
                    "versions": {
                        name: version(name)
                        for name in ("relarena", "relbench", "numpy")
                    },
                },
            }
            self._add_split(
                result,
                split,
                split.eval_target,
                trial.val_pred,
                trial.val_metrics,
                trial.fit_time_tuning,
                trial.predict_time_tuning,
                False,
            )
            root = self._root(trial)
            if (root / "results.pkl").exists():
                raise FileExistsError(root / "results.pkl")
            _save_result(root / "validation.partial", result)

    def save_test(
        self,
        trial: TrialResult,
        result: dict[str, Any],
        split: OuterSplit,
        *,
        additional: bool,
    ) -> None:
        """Complete a config result with test predictions, labels and refit timings.

        TabArena's results.pkl schema uses metric errors and per-config training
        and inference times. Separate tuning/refit timings and the additional-refit
        flag remain in relarena metadata for benchmark runtime accounting.
        """
        root = self._root(trial)
        validation_path = root / "validation.partial"
        with validation_path.open("rb") as stream:
            artifact = pickle.load(stream)
        self._add_split(
            artifact,
            split,
            self.source.task.get_table("test", mask_input_cols=False),
            result["test_pred"],
            result["test_metrics"],
            result["fit_time_refit"],
            result["predict_time_refit"],
            additional,
        )
        artifact.update(
            metric_error=get_metric(self.source.metric).to_error(result["test_score"]),
            time_train_s=trial.fit_time_tuning + result["fit_time_refit"],
            time_infer_s=result["predict_time_refit"],
        )
        _save_result(root / "results.pkl", artifact)
        validation_path.unlink()
