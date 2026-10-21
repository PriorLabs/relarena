"""Fit supervised RelFlow models and select exact full-validation checkpoints.

Imported only when the RelFlow system runs. No callback reads the masked outer
table, and the scalar prediction writer preserves the anchor row axis.
"""

from __future__ import annotations

import copy
import gc
import logging
import math
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import relflow as rf
import torch
import yaml
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def pin_text(model: dict[str, Any]) -> dict[str, Any]:
    """Resolve the immutable frozen encoder once and bind its local snapshot."""
    from huggingface_hub import snapshot_download

    result = copy.deepcopy(model)
    pending = [result]
    path = None
    while pending:
        node = pending.pop()
        if node.get("type") == "text":
            if node.get("model") != TEXT_MODEL:
                raise ValueError(f"Unpinned RelFlow text model: {node.get('model')!r}")
            if path is None:
                path = snapshot_download(
                    repo_id=TEXT_MODEL,
                    revision=TEXT_REVISION,
                    allow_patterns=[
                        "config.json",
                        "model.safetensors",
                        "tokenizer*",
                        "special_tokens_map.json",
                        "vocab.txt",
                    ],
                )
            node["model"] = path
        pending.extend(node.get("fields", {}).values())
    return result


def reset_text_cache(model: rf.Model) -> None:
    """Use cold per-field caches for validation and checkpoint reload parity."""
    from relflow.tensorfields.extensions.text import Embedder, EmbeddingCache

    for address in model.schema.requests:
        embedder = model.nodes[address].embedder
        if isinstance(embedder, Embedder) and embedder.cache is not None:
            embedder.cache = EmbeddingCache(embedder.cache.capacity)


def batches(path: Path, *, predict: bool = False) -> Iterator[pa.RecordBatch]:
    """Open a restartable bounded Parquet stream, removing labels at inference."""
    with pq.ParquetFile(path) as source:
        columns = source.schema_arrow.names
        if predict:
            columns = [name for name in columns if name != "target"]
        yield from source.iter_batches(
            batch_size=16_384, columns=columns, use_threads=False
        )


def device_options() -> dict[str, Any]:
    """Run on one CUDA GPU when present, with a portable CPU test fallback."""
    cuda = torch.cuda.is_available()
    return {
        "accelerator": "gpu" if cuda else "cpu",
        "devices": 1,
        "precision": (
            "bf16-mixed" if cuda and torch.cuda.is_bf16_supported() else "32-true"
        ),
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def score(kind: str, labels: np.ndarray, values: np.ndarray) -> float:
    """Compute full-split AUROC or source-unit MAE, never batch averages."""
    if (
        labels.ndim != 1
        or values.shape != labels.shape
        or not len(labels)
        or not np.isfinite(labels).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("RelFlow validation needs aligned, nonempty finite vectors")
    if kind == "binary_classification":
        if set(np.unique(labels)) != {0, 1} or ((values < 0) | (values > 1)).any():
            raise ValueError("RelFlow AUROC requires both classes and probabilities")
        return float(roc_auc_score(labels, values))
    if kind != "regression":
        raise ValueError(f"Unknown RelFlow task type: {kind!r}")
    return float(np.mean(np.abs(values - labels)))


class Selection(lit.Callback):
    """Keep the best complete validation checkpoint and stop at safe boundaries."""

    def __init__(
        self,
        recipe: dict[str, Any],
        train_rows: int,
        validation_rows: int,
        checkpoint: Path,
        deadline: float,
    ) -> None:
        """Bind selection to one task, complete row axes and runtime allocation."""
        super().__init__()
        self.kind = recipe["contract"]["task_type"]
        self.settings = recipe["training"]
        self.validation_rows = validation_rows
        self.checkpoint = checkpoint
        self.deadline = deadline
        batch_size = recipe["model"]["batch_size"]
        self.minimum_steps = max(
            self.settings["min_steps"],
            math.ceil(train_rows / batch_size) * self.settings["min_full_passes"],
        )
        self.best: float | None = None
        self.stale = 0
        self.last_step = -1
        self.current: np.ndarray | None = None
        self.active = False
        self.positions: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.hook: Any = None

    def on_fit_start(self, trainer: lit.Trainer, pl_module: rf.Model) -> None:
        """Observe the existing scalar decoder without a second forward pass."""
        self.hook = pl_module.nodes[rf.Address("target")].decoder.register_forward_hook(
            self.capture
        )

    def capture(self, module: Any, inputs: Any, output: Any) -> None:
        """Retain this validation batch's scalar prediction in source units."""
        if not self.active:
            return
        if self.current is not None:
            raise RuntimeError("RelFlow target decoded more than once in one batch")
        value = output.payload["content"].detach().reshape(-1)
        if self.kind == "binary_classification":
            value = value.sigmoid()
        self.current = value.float().cpu().numpy().astype(np.float64)

    def on_validation_epoch_start(
        self, trainer: lit.Trainer, pl_module: rf.Model
    ) -> None:
        """Discard transient encoder history before the complete validation pass."""
        reset_text_cache(pl_module)
        self.active = True
        self.positions.clear()
        self.labels.clear()
        self.values.clear()
        self.current = None

    def on_validation_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: rf.Model,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Pair decoder outputs with retained source positions and target labels."""
        if self.current is None or len(self.current) != len(batch.source):
            raise RuntimeError(
                "RelFlow validation predictions lost the source row axis"
            )
        self.positions.append(batch.source["__row_position"].to_numpy())
        self.labels.append(batch.source["target"].to_numpy().astype(np.float64))
        self.values.append(self.current)
        self.current = None

    def on_validation_epoch_end(
        self, trainer: lit.Trainer, pl_module: rf.Model
    ) -> None:
        """Select only after checking that every validation row occurred once."""
        self.active = False
        positions = np.concatenate(self.positions) if self.positions else np.array([])
        if len(positions) != self.validation_rows or not np.array_equal(
            np.sort(positions), np.arange(self.validation_rows)
        ):
            raise RuntimeError("RelFlow selection requires every validation row once")
        value = score(
            self.kind, np.concatenate(self.labels), np.concatenate(self.values)
        )
        improved = self.consider(pl_module, value, int(trainer.global_step))
        if improved:
            self.stale = 0
        elif trainer.global_step >= self.minimum_steps:
            self.stale += 1
        if (
            trainer.global_step >= self.minimum_steps
            and self.stale >= self.settings["patience"]
        ):
            trainer.should_stop = True

    def consider(self, model: rf.Model, value: float, step: int) -> bool:
        """Save an improved scalar objective with its complete current schema."""
        if not math.isfinite(value):
            raise ValueError("RelFlow checkpoint selection requires a finite metric")
        delta = self.settings["min_delta"]
        improved = self.best is None or (
            value > self.best + delta
            if self.kind == "binary_classification"
            else value < self.best - delta
        )
        self.last_step = step
        if improved:
            self.best = value
            model.save(self.checkpoint)
        logger.info(
            "RelFlow validation step=%d score=%g best=%s", step, value, self.best
        )
        return improved

    def on_train_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: rf.Model,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """End training safely after an optimizer update when its budget expires."""
        if time.monotonic() >= self.deadline:
            trainer.should_stop = True

    def teardown(self, trainer: lit.Trainer, pl_module: rf.Model, stage: str) -> None:
        """Detach the prediction hook when Lightning closes this training run."""
        self.active = False
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def predict(model: rf.Model, path: Path, *, seed: int) -> np.ndarray:
    """Write probabilities/source-unit numbers in the original anchor row order."""
    rows = pq.ParquetFile(path).metadata.num_rows
    if not rows:
        return np.empty(0, dtype=np.float64)
    reset_text_cache(model)
    module = rf.ArrowDataModule(
        predict=partial(batches, path, predict=True),
        seed=seed,
        shuffle=False,
        num_workers=0,
        retain=("__row_position",),
    )
    output = model.predict(datamodule=module, **device_options())
    positions = pc.struct_field(output["inputs"], "__row_position").to_numpy()
    content = pc.struct_field(
        pc.struct_field(output["predictions"], "target"), "content"
    )
    binary = pa.types.is_struct(content.type)
    if binary:
        content = pc.struct_field(content, "probability")
    values = content.to_numpy().astype(np.float64)
    if (
        len(values) != rows
        or not np.array_equal(np.sort(positions), np.arange(rows))
        or not np.isfinite(values).all()
        or (binary and ((values < 0) | (values > 1)).any())
    ):
        raise RuntimeError(
            "RelFlow predictions must be finite and cover each anchor once"
        )
    return values[np.argsort(positions)]


def train(
    train_path: Path,
    validation_path: Path,
    recipe: dict[str, Any],
    directory: Path,
    *,
    seed: int,
    deadline: float = math.inf,
) -> rf.Model:
    """Fit, verify a cold reload, and return the best full-validation checkpoint."""
    lit.seed_everything(seed, workers=True)
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.yaml"
    model_path.write_text(yaml.safe_dump(pin_text(recipe["model"]), sort_keys=False))
    model = rf.Model.from_yaml(model_path)
    if model.schema.objectives != [rf.Address("target")]:
        raise ValueError("RelFlow recipes must have exactly one supervised target")
    train_rows = pq.ParquetFile(train_path).metadata.num_rows
    labels = pq.read_table(validation_path, columns=["target"])["target"].to_numpy()
    if not train_rows or not len(labels):
        raise ValueError("RelFlow training and validation splits must be nonempty")
    selection = Selection(
        recipe, train_rows, len(labels), directory / "best.ckpt", deadline
    )
    data = rf.ArrowDataModule(
        train=partial(batches, train_path),
        validate=partial(batches, validation_path),
        seed=seed,
        shuffle={"train": True, "validate": False},
        shuffle_rows=16_384,
        num_workers=0,
        retain=("__row_position",),
    )
    settings = recipe["training"]
    optimizer = dict(recipe["optimizer"])
    if not torch.cuda.is_available():
        optimizer["fused"] = False
    model.fit(
        data,
        **optimizer,
        callbacks=[selection],
        early_stopping=False,
        rollback=False,
        max_epochs=settings["max_epochs"],
        max_steps=settings["max_steps"],
        gradient_clip_val=settings["gradient_clip_val"],
        deterministic=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        val_check_interval=min(
            settings["val_every"], math.ceil(train_rows / model.batch_size)
        ),
        check_val_every_n_epoch=None,
        **device_options(),
    )
    step = int(model.trainer.global_step)
    if step < 1:
        raise RuntimeError("RelFlow training completed without an optimizer update")
    if selection.last_step != step:
        value = score(
            selection.kind, labels, predict(model, validation_path, seed=seed)
        )
        selection.consider(model, value, step)
    if selection.best is None:
        raise RuntimeError(
            "RelFlow training produced no validation-selected checkpoint"
        )
    del model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restored = rf.Model.load(selection.checkpoint)
    value = score(selection.kind, labels, predict(restored, validation_path, seed=seed))
    if not np.isclose(value, selection.best, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            f"RelFlow checkpoint reload changed validation metric: "
            f"selected={selection.best}, restored={value}"
        )
    parameters = sum(p.numel() for p in restored.parameters() if p.requires_grad)
    logger.info("RelFlow selected score=%g trained_parameters=%d", value, parameters)
    return restored
