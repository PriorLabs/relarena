"""Fixed shared relflow recipes, resolved only from original training-set size.

The harness owns both temporal splits and final scoring. This adapter performs
no HPO and never reloads benchmark data or reads hidden test labels.
"""

from __future__ import annotations

import copy
import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from relbench.base import Database, EntityTask, Table, TaskType

from relarena_core.model import RelArenaModel
from relarena_core.registry import register_model, registry
from relarena_core.search_space import SearchSpace, TaskStats


def search_space(stats: TaskStats) -> SearchSpace:
    """Return one frozen default, carrying its original tier through the refit."""
    tier = (
        "small"
        if stats.num_train_nodes < 20_000
        else "medium"
        if stats.num_train_nodes < 500_000
        else "large"
    )
    recipes = json.loads(Path(__file__).with_name("recipes.json").read_text())
    config = copy.deepcopy(recipes[tier])
    config["training_rows"] = stats.num_train_nodes
    return SearchSpace(default_overrides=config)


@register_model(search_space=search_space)
class RelFlowModel(RelArenaModel):
    """Fit the fixed size-tier recipe, using only supplied censored data."""

    name = "relflow-2026-11-04"
    refit_on_full_data = True

    def execution(self) -> AbstractContextManager[None]:
        """Use the numerical defaults of the published entry."""
        return nullcontext()

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Derive the schema and train the resolved recipe on the supplied split."""
        from .training import fit

        with self.execution():
            self.fitted = fit(
                task,
                db,
                train_table,
                self.config,
                seed=seed,
                time_limit=time_limit,
                phase=self.run_identity.phase if self.run_identity else None,
                namespace="database"
                if self.run_identity is None
                else self.run_identity.dataset,
            )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Return one finite scalar or positive-class probability per input row."""
        from .training import predict

        if not hasattr(self, "fitted"):
            raise RuntimeError("Fit the relflow model before predicting.")
        with self.execution():
            return predict(self.fitted, task, db, table)


def search_space_053(stats: TaskStats) -> SearchSpace:
    """Freeze one recipe using the original inner training-anchor count."""
    rows = stats.num_train_nodes
    tier = (
        "small-a" if rows < 10_000 else "small-b" if rows < 20_000 else "medium-large"
    )
    recipes = json.loads(Path(__file__).with_name("recipes-0.5.3.json").read_text())
    return SearchSpace(default_overrides={**recipes[tier], "training_rows": rows})


@register_model(search_space=search_space_053)
class RelFlow053Model(RelFlowModel):
    """Model mature outcomes and aggregates with a fixed recipe per size tier."""

    name = "relflow-2026-11-25"
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        """Bind direct and registry callers to the relflow 0.5.3 schema."""
        super().__init__({**config, "schema_version": 6}, **kwargs)

    def execution(self) -> AbstractContextManager[None]:
        """Use the same single-device kernel settings for fitting and prediction."""
        from .runtime import numerical_settings

        return numerical_settings()


registry.register_alias("relflow-latest", RelFlow053Model.name)
