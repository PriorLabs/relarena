"""Native relflow fit/predict execution for fixed size-tier recipes.

The registered adapters load the neural backend when fitting or predicting.
The harness supplies the phase database, label tables and execution budget.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import relflow as rf
import torch
import yaml
from relbench.base import Database, EntityTask, Table

from . import raw, targets
from .context import Context, infer, materialize

if TYPE_CHECKING:
    from .history import History


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


def encoder_snapshot(identifier: str, *, individual_files: bool) -> str:
    """Resolve the exact weight and tokenizer files during the timed fit."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if identifier != ENCODER:
        raise ValueError(
            f"relflow has no frozen revision for text encoder {identifier!r}."
        )
    if not individual_files:
        return snapshot_download(
            repo_id=identifier,
            revision=ENCODER_REVISION,
            allow_patterns=list(ENCODER_FILES),
        )
    # A file pinned to its commit can resolve locally without listing the remote tree.
    files = [
        hf_hub_download(repo_id=identifier, filename=name, revision=ENCODER_REVISION)
        for name in ENCODER_FILES
    ]
    return str(Path(files[0]).parent)


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
    history: History | None = None
    prediction_deadline: float | None = None


def schema(
    structure: raw.Task,
    config: dict[str, Any],
    *,
    store: Path | None = None,
    history: History | None = None,
) -> dict[str, Any]:
    """Apply independent root, branch and target settings from the frozen recipe."""
    if config.get("schema_version") == 6 and history is None:
        raise ValueError("The relflow 0.5.3 schema requires fitted outcome history.")
    model = raw.model(structure)
    if history is not None:
        model["fields"].update(history.fields(structure))
    recipe = config["recipe"]
    root_cross = recipe.get("root_cross", False)
    branch_cross = recipe.get("branch_cross", False)
    if type(root_cross) is not bool or type(branch_cross) is not bool:
        raise ValueError("root_cross and branch_cross must be booleans.")
    if (root_cross or branch_cross) and "cross" not in getattr(
        rf.Branch, "model_fields", {}
    ):
        raise RuntimeError("Field crossing requires a relflow SDK with Branch.cross.")
    if root_cross:
        model["cross"] = True
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
                if branch_cross:
                    field["cross"] = True
                field.update(
                    {
                        "length": config["caps"]["outer" if depth == 0 else "inner"],
                        "overflow": "random",
                        "n_layers": recipe["branch_layers"],
                        "n_heads": recipe["branch_heads"],
                        "ffnx": recipe["branch_ffnx"],
                        "dropout": recipe["branch_dropout"],
                        "mask": recipe["branch_mask"],
                        "reduction": {
                            "type": "attention",
                            "n_outputs": recipe["branch_outputs"],
                        },
                    }
                )
                if history is not None and depth == 0 and name == history.history:
                    field.update(length=None, overflow=None)
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
                if field["type"] in {"number", "ordinal"}:
                    field["objective"] = recipe["objective"]
                if (
                    field["type"] in {"number", "ordinal", "boolean"}
                    and recipe.get("ensemble") is not None
                ):
                    field["ensemble"] = recipe["ensemble"]
            else:
                field["mask"] = recipe["mask"]
                if field["type"] == "text":
                    field["max_length"] = (
                        min(128, config.get("text_token_cap", 128))
                        if history is not None
                        else config.get("text_token_cap", field["max_length"])
                    )
                    identifier = field["model"]
                    if identifier not in encoders:
                        encoders[identifier] = encoder_snapshot(
                            identifier, individual_files=history is not None
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
                    "duration",
                    "ordinal",
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
    history: History | None = None,
) -> Any:
    """Keep source buffers and worker queues bounded by the tier settings."""
    split = "predict" if predict else "train"
    workers = settings["workers"] if torch.cuda.is_available() else 0
    options = {}
    if not predict and history is not None and history.hurdle is not None:
        from .hurdle import positive_target

        options["preprocessor"] = {"train": positive_target}
    return rf.ArrowDataModule(
        **options,
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
    phase: str | None = None,
) -> Fitted:
    """Fit all available labels, retaining validation anchors' frozen history."""
    use_outcome_history = config.get("schema_version") == 6
    if len(train_table) == 0:
        raise ValueError("relflow requires at least one training anchor.")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("relflow fit time_limit must be finite and positive.")
    started = time.monotonic()
    cutoff = task.dataset.val_timestamp
    stale = train_table.df[task.time_col] >= cutoff
    if use_outcome_history and phase is None:
        phase = "outer" if stale.any() else "inner"
    deadline = None if time_limit is None else started + time_limit
    hard_budget = (
        config.get("outer_budget", {}).get("hard_s") if phase == "outer" else None
    )
    if hard_budget is not None:
        deadline = min(deadline or math.inf, started + hard_budget)
    scratch = tempfile.TemporaryDirectory(prefix="relarena-relflow-")
    directory = Path(scratch.name)
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
        frame = train_table.df
        history = None
        if use_outcome_history:
            labels = pa.Array.from_pandas(frame[task.target_col])
            structure = replace(
                structure, target_kind=targets.kind(task.task_type, labels)
            )
            from .history import History

            history = History.fit(
                context,
                task,
                train_table,
                structure,
                phase,
                ordinal_hurdle=config["recipe"].get("ordinal_hurdle", True),
                number_hurdle=config["recipe"].get("number_hurdle", True),
            )
            context.freeze()
            history.prepare(context)
        # Intentional recency bias: every source pass ends with validation-period
        # anchors. The bounded loader shuffle preserves this bias in the EMA.
        # Their histories stay frozen at the validation boundary.
        parts = []
        for name, selected in (
            ("fresh", frame.loc[~stale].sample(frac=1, random_state=seed)),
            ("stale", frame.loc[stale]),
        ):
            if name == "stale" and history is None:
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
                    history=history,
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
            schema(structure, config, store=directory / "text-store", history=history),
            sort_keys=False,
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
    steps = math.ceil(len(train_table) / batch_size) * recipe["passes"]
    from .runtime import RuntimeGuard, scheduler

    threshold = recipe.get("gradient_clip_val", 0.25)
    floor = recipe.get("lr_floor_ratio", 1.0)
    callbacks = []
    if hard_budget is not None:
        callbacks.append(
            RuntimeGuard(
                started,
                steps,
                math.ceil(len(train_table) / batch_size),
                config["outer_budget"],
            )
        )
    if recipe["ema_decay"] is not None:
        callbacks.append(rf.EMA(decay=recipe["ema_decay"]))
    model.fit(
        loader(
            model,
            directory / "train.parquet",
            seed,
            settings,
            predict=False,
            history=history,
        ),
        learning_rate=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
        fused=cuda,
        scheduler=scheduler(steps, floor),
        callbacks=callbacks,
        early_stopping=False,
        rollback=False,
        max_epochs=-1,
        max_steps=steps,
        max_time=None if remaining is None else timedelta(seconds=remaining),
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=threshold,
        gradient_clip_algorithm="norm",
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
        raise TimeoutError("relflow did not finish its declared full-pass budget.")
    model.eval()
    return Fitted(
        model,
        structure,
        scratch,
        seed,
        accelerator,
        precision,
        batch_size,
        settings,
        history,
        started + hard_budget if hard_budget is not None else None,
    )


def predict(fitted: Fitted, task: EntityTask, db: Database, table: Table) -> np.ndarray:
    """Materialize the current provided DB and restore the exact anchor order."""
    if len(table) == 0:
        return np.empty(0, dtype=np.float64)
    from .runtime import PredictionDeadline

    clock = (
        PredictionDeadline(fitted.prediction_deadline)
        if fitted.prediction_deadline is not None
        else None
    )
    if clock is not None:
        clock.check()
    directory = Path(fitted.directory.name)
    settings = fitted.execution
    context = Context.open(
        db,
        fitted.history.boundary
        if fitted.history is not None
        else task.dataset.val_timestamp,
        directory,
        threads=settings["duckdb_threads"],
        memory_limit=settings["duckdb_memory_limit"],
        max_temp_directory_size=settings["duckdb_spill_limit"],
    )
    try:
        if fitted.history is not None:
            context.freeze()
            fitted.history.prepare(context)
        path = materialize(
            context,
            fitted.structure,
            task,
            table,
            directory / "predict.parquet",
            labelled=False,
            deadline=clock.deadline if clock else None,
            history=fitted.history,
        )
    finally:
        context.close()
    output = fitted.model.predict(
        loader(fitted.model, path, fitted.seed, fitted.execution, predict=True),
        batch_size=min(64, fitted.batch_size),
        accelerator=fitted.accelerator,
        devices=1,
        precision=fitted.precision,
        callbacks=[clock] if clock else [],
        logger=False,
        enable_progress_bar=False,
    )
    if fitted.history is not None and fitted.history.hurdle is not None:
        decoded = fitted.history.hurdle.postprocessor(fitted.model).run(output)
        positions = decoded["inputs"].struct.field("__row_position").to_numpy()
        values = decoded["estimate"].to_numpy()
    else:
        result = output if isinstance(output, pa.Table) else output.to_arrow()
        positions = pc.struct_field(result["inputs"], "__row_position").to_numpy()
        content = pc.struct_field(
            pc.struct_field(result["predictions"], "target"), "content"
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
    if clock is not None:
        clock.check()
    return values[np.argsort(positions)]
