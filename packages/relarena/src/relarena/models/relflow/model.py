"""Run frozen RelFlow recipes through RelArena's censored split interface.

The system selects a checkpoint using the complete inner validation split. It
keeps that train-only checkpoint and predicts on the masked outer table using
the supplied outer database. It never downloads a database or reads test labels.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np
from relbench.base import EntityTask, TaskType

from relarena.models.relflow.config import load_recipe
from relarena_core.dataset import InnerSplit, OuterSplit
from relarena_core.registry import register_system, registry
from relarena_core.system import RelArenaSystem


def check_budget(deadline: float, phase: str) -> None:
    """Refuse a new phase once the caller's end-to-end soft budget is exhausted."""
    if time.monotonic() >= deadline:
        raise TimeoutError(f"RelFlow runtime budget exhausted before {phase}")


@register_system
class RelFlowSystem(RelArenaSystem):
    """Schema-defined supervised models with validation-selected task recipes."""

    name = "relflow-2026-10-21"
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        """Select on validation and return ordered, label-free outer predictions.

        RelArena's run identity chooses one of the 21 published recipes. The
        runtime budget starts before optional imports and cold preparation;
        training reserves the final fifth for checkpoint verification and
        outer preparation/prediction. Limits stop safely at batch/phase
        boundaries rather than killing an update or returning partial rows.
        """
        started = time.monotonic()
        if time_limit is not None and (
            not math.isfinite(time_limit) or time_limit <= 0
        ):
            raise ValueError("RelFlow time_limit must be finite and positive")
        budget = 24 * 60 * 60 if time_limit is None else min(time_limit, 24 * 60 * 60)
        deadline = started + budget
        identity = self.run_identity
        if identity is None or not identity.task:
            raise ValueError(
                "RelFlow requires a RunIdentity with dataset and task names; "
                "use RelArena's experiment runner"
            )
        recipe = load_recipe(identity.dataset, identity.task)
        contract = recipe["contract"]
        attributes = {
            "entity_table": "entity_table",
            "entity_col": "anchor_entity_column",
            "time_col": "timestamp_column",
            "target_col": "target_column",
        }
        for attribute, key in attributes.items():
            if getattr(task, attribute) != contract[key]:
                raise ValueError(
                    f"RelFlow recipe {identity.dataset}/{identity.task}: "
                    f"{attribute} must be {contract[key]!r}"
                )
        expected = (
            TaskType.BINARY_CLASSIFICATION
            if contract["task_type"] == "binary_classification"
            else TaskType.REGRESSION
        )
        if task.task_type != expected:
            raise ValueError(f"RelFlow recipe requires task type {expected}")
        if task.target_col in outer_split.eval_table.df.columns:
            raise ValueError("RelFlow outer evaluation table must have masked labels")

        # Optional imports stay out of discovery, including imports of modules
        # that themselves own Lightning callbacks or Arrow/DuckDB operations.
        from relarena.models.relflow.data import prepare, snapshot
        from relarena.models.relflow.training import predict, train

        with tempfile.TemporaryDirectory(prefix="relarena-relflow-") as temporary:
            directory = Path(temporary)
            check_budget(deadline, "inner preparation")
            inner = snapshot(
                inner_split.db_state,
                directory / "inner",
                sources=contract["sources"],
            )
            train_path = prepare(
                inner,
                inner_split.train_table,
                task,
                recipe["query"],
                directory / "inner" / "train",
                include_target=True,
            )
            check_budget(deadline, "validation preparation")
            validation_path = prepare(
                inner,
                inner_split.eval_table,
                task,
                recipe["query"],
                directory / "inner" / "validate",
                include_target=True,
            )
            check_budget(deadline, "training")
            training_deadline = started + budget * 0.8
            if time.monotonic() >= training_deadline:
                raise TimeoutError(
                    "RelFlow preparation exhausted its training allocation"
                )
            model = train(
                train_path,
                validation_path,
                recipe,
                directory / "fit",
                seed=seed,
                deadline=training_deadline,
            )
            check_budget(deadline, "outer preparation")
            outer = snapshot(
                outer_split.db_state,
                directory / "outer",
                sources=contract["sources"],
            )
            prediction_path = prepare(
                outer,
                outer_split.eval_table,
                task,
                recipe["query"],
                directory / "outer" / "predict",
                include_target=False,
            )
            check_budget(deadline, "outer prediction")
            return predict(model, prediction_path, seed=seed)


registry.register_alias("relflow-latest", RelFlowSystem.name)
