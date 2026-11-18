"""Native relflow fit/predict execution for fixed size-tier recipes.

Only this execution module imports relflow and its optional neural dependencies.
No experiment controller, test evaluator or external data source is included.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
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
from relbench.base import Database, EntityTask, Table

from . import raw
from .context import Context, infer, materialize

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


@dataclass
class Fitted:
    """Trained model, fixed structure and owned scratch directory."""

    model: Any
    structure: raw.Task
    directory: Any
    seed: int
    accelerator: str
    precision: str
    batch_size: int
    execution: dict[str, Any]


def schema(
    structure: raw.Task, config: dict[str, Any], *, store: Path | None = None
) -> dict[str, Any]:
    """Apply independent root, branch and target settings from the frozen recipe."""
    model = raw.model(structure)
    recipe = config["recipe"]
    encoders: dict[str, str] = {}
    model.update(
        {
            key: recipe[key]
            for key in ("d_model", "n_layers", "n_heads", "ffnx", "dropout")
        }
    )

    def configure(fields: dict[str, Any], depth: int) -> None:
        for name, field in fields.items():
            field["ffnx"] = recipe["ffnx"]
            if field["type"] == "branch":
                field.update(
                    {
                        "length": config["caps"]["outer" if depth == 0 else "inner"],
                        "overflow": "random",
                        "n_layers": recipe["branch_layers"],
                        "n_heads": recipe["branch_heads"],
                        "ffnx": recipe["branch_ffnx"],
                        "dropout": 0.0,
                        "mask": recipe["branch_mask"],
                        "reduction": {
                            "type": "attention",
                            "n_outputs": recipe["branch_outputs"],
                        },
                    }
                )
                configure(field["fields"], depth + 1)
            elif name == "target":
                field.update(
                    {
                        "n_linear": recipe["decoder_layers"],
                        "n_heads": recipe["decoder_heads"],
                        "ffnx": recipe["decoder_ffnx"],
                        "dropout": 0.0,
                    }
                )
                if field["type"] == "number":
                    field["objective"] = recipe["objective"]
            else:
                field["mask"] = recipe["mask"]
                if field["type"] == "text":
                    identifier = field["model"]
                    if identifier not in encoders:
                        from huggingface_hub import snapshot_download

                        if identifier != ENCODER:
                            raise ValueError(
                                "relflow has no frozen revision for text encoder "
                                f"{identifier!r}."
                            )
                        # The local snapshot fixes weights and tokenizer together;
                        # acquiring it belongs to this fit's preparation clock.
                        encoders[identifier] = snapshot_download(
                            repo_id=identifier,
                            revision=ENCODER_REVISION,
                            allow_patterns=list(ENCODER_FILES),
                        )
                    field["model"] = encoders[identifier]
                    if store is not None:
                        field["store"] = str(store)
                        field["cache"] = config["execution"]["text_token_cache"]
                        field["resident"] = config["execution"]["text_resident_mib"]
                if recipe["jitter"] and field["type"] in {
                    "number",
                    "vector",
                    "dateparts",
                    "text",
                }:
                    field["jitter"] = {"add": recipe["jitter"]}

    configure(model["fields"], 0)
    return model


def batches(path: Path) -> Iterator[pa.RecordBatch]:
    """Reopen the source in bounded record batches for each loader pass."""
    yield from pq.ParquetFile(path).iter_batches(batch_size=8192, use_threads=False)


def loader(
    model: Any,
    path: Path,
    seed: int,
    settings: dict[str, Any],
    *,
    predict: bool,
) -> Any:
    """Keep source buffers and worker queues bounded by the tier settings."""
    split = "predict" if predict else "train"
    workers = settings["workers"] if torch.cuda.is_available() else 0
    return rf.ArrowDataModule(
        model=model,
        **{split: partial(batches, path)},
        seed=seed,
        shuffle={split: not predict},
        shuffle_rows=settings["shuffle_rows"],
        num_workers=workers,
        persistent_workers=bool(workers and settings["persistent_workers"]),
        prefetch_factor=settings["prefetch_factor"],
        multiprocessing_context="spawn" if workers else None,
        pin_memory=False,
        retain=("__row_position",),
    )


def fit(
    task: EntityTask,
    db: Database,
    train_table: Table,
    config: dict[str, Any],
    *,
    seed: int,
    time_limit: float | None,
    namespace: str = "database",
) -> Fitted:
    """Fit all available labels, retaining validation anchors' frozen history."""
    if len(train_table) == 0:
        raise ValueError("relflow requires at least one training anchor.")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("relflow fit time_limit must be finite and positive.")
    started = time.monotonic()
    deadline = None if time_limit is None else started + time_limit
    scratch = tempfile.TemporaryDirectory(prefix="relarena-relflow-")
    directory = Path(scratch.name)
    cutoff = task.dataset.val_timestamp
    settings = config["execution"]
    context = Context.open(
        db,
        cutoff,
        directory,
        threads=settings["duckdb_threads"],
        memory_limit=settings["duckdb_memory_limit"],
        max_temp_directory_size=settings["duckdb_spill_limit"],
    )
    try:
        structure = infer(context, task, config, namespace=namespace)
        # Intentional recency bias: every source pass ends with validation-period
        # anchors. The bounded loader shuffle preserves this bias in the EMA.
        # Their histories stay frozen at the validation boundary.
        frame = train_table.df
        stale = frame[task.time_col] >= cutoff
        parts = []
        for name, selected in (
            ("fresh", frame.loc[~stale].sample(frac=1, random_state=seed)),
            ("stale", frame.loc[stale]),
        ):
            if name == "stale":
                context.freeze()
            if len(selected):
                path = materialize(
                    context,
                    structure,
                    task,
                    Table(
                        selected,
                        train_table.fkey_col_to_pkey_table,
                        pkey_col=train_table.pkey_col,
                        time_col=train_table.time_col,
                    ),
                    directory / f"{name}.parquet",
                    labelled=True,
                    deadline=deadline,
                )
                parts.append(path)
        writer, offset = None, 0
        try:
            for path in parts:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                    table = pa.Table.from_batches([batch])
                    table = table.set_column(
                        table.schema.get_field_index("__row_position"),
                        "__row_position",
                        pa.array(
                            np.arange(offset, offset + len(table), dtype=np.int64)
                        ),
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(
                            directory / "train.parquet", table.schema
                        )
                    writer.write_table(table)
                    offset += len(table)
        finally:
            if writer is not None:
                writer.close()
    finally:
        context.close()
    schema_path = directory / "model.yaml"
    schema_path.write_text(
        yaml.safe_dump(
            schema(structure, config, store=directory / "text-store"), sort_keys=False
        )
    )
    cuda = torch.cuda.is_available()
    accelerator = "gpu" if cuda else "cpu"
    precision = "bf16-mixed" if cuda and torch.cuda.is_bf16_supported() else "32-true"
    recipe = config["recipe"]
    settings = config["execution"]
    batch_size = settings["batch_size"]
    lit.seed_everything(seed, workers=True)
    model = rf.Model.from_yaml(schema_path, batch_size=batch_size)
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("relflow exhausted its fit budget in preparation.")
    steps = math.ceil(len(train_table) / batch_size) * 8
    model.fit(
        loader(model, directory / "train.parquet", seed, settings, predict=False),
        learning_rate=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
        fused=cuda,
        callbacks=[rf.EMA(decay=recipe["ema_decay"])],
        early_stopping=False,
        rollback=False,
        max_epochs=-1,
        max_steps=steps,
        max_time=None if remaining is None else timedelta(seconds=remaining),
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=0.25,
        deterministic=True,
        default_root_dir=str(directory),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        log_every_n_steps=50,
    )
    if model.trainer.global_step != steps:
        raise TimeoutError("relflow did not finish its fixed eight-pass budget.")
    model.eval()
    return Fitted(
        model, structure, scratch, seed, accelerator, precision, batch_size, settings
    )


def predict(fitted: Fitted, task: EntityTask, db: Database, table: Table) -> np.ndarray:
    """Materialize the current provided DB and restore the exact anchor order."""
    if len(table) == 0:
        return np.empty(0, dtype=np.float64)
    directory = Path(fitted.directory.name)
    settings = fitted.execution
    context = Context.open(
        db,
        task.dataset.val_timestamp,
        directory,
        threads=settings["duckdb_threads"],
        memory_limit=settings["duckdb_memory_limit"],
        max_temp_directory_size=settings["duckdb_spill_limit"],
    )
    try:
        path = materialize(
            context,
            fitted.structure,
            task,
            table,
            directory / "predict.parquet",
            labelled=False,
        )
    finally:
        context.close()
    output = fitted.model.predict(
        loader(fitted.model, path, fitted.seed, fitted.execution, predict=True),
        batch_size=min(64, fitted.batch_size),
        accelerator=fitted.accelerator,
        devices=1,
        precision=fitted.precision,
        logger=False,
        enable_progress_bar=False,
    )
    positions = pc.struct_field(output["inputs"], "__row_position").to_numpy()
    content = pc.struct_field(
        pc.struct_field(output["predictions"], "target"), "content"
    )
    if pa.types.is_struct(content.type):
        content = pc.struct_field(content, "probability")
    values = content.to_numpy().astype(np.float64)
    if len(values) != len(table) or not np.array_equal(
        np.sort(positions), np.arange(len(table))
    ):
        raise RuntimeError("relflow must return exactly one prediction per anchor.")
    if not np.isfinite(values).all():
        raise RuntimeError("relflow predictions must be finite.")
    return values[np.argsort(positions)]
