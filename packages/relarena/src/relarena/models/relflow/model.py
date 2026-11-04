"""Fixed shared relflow recipes, resolved only from original training-set size.

The harness owns both temporal splits and final scoring. This adapter performs
no HPO and never reloads benchmark data or reads hidden test labels.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from relbench.base import Database, EntityTask, Table

from relarena_core.identity import RunIdentity
from relarena_core.model import RelArenaModel
from relarena_core.registry import register_model
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


def execution(config: dict[str, Any], identity: RunIdentity | None) -> dict[str, Any]:
    """Freeze measured batches without selecting architecture or scores by task."""
    result = dict(config["execution"])
    # These are the effective batches used in the original singleton runs.
    # They alter neither the tier recipe nor its history geometry.
    if (
        identity is not None
        and config["tier"] == "medium"
        and (identity.dataset, identity.task)
        in {
            ("rel-avito", "user-clicks"),
            ("rel-avito", "user-visits"),
            ("rel-trial", "site-success"),
        }
    ):
        result["batch_size"] = 256
    return result


@register_model(search_space=search_space)
class RelFlowModel(RelArenaModel):
    """Fit the original singleton recipe, using only supplied censored data."""

    name = "relflow-2026-11-04"
    refit_on_full_data = True

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
        """Derive label-free structure and fit eight full passes at constant LR."""
        from .training import fit

        self.fitted = fit(
            task,
            db,
            train_table,
            {**self.config, "execution": execution(self.config, self.run_identity)},
            seed=seed,
            time_limit=time_limit,
            namespace="database"
            if self.run_identity is None
            else self.run_identity.dataset,
        )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Return one finite scalar or positive-class probability per input row."""
        from .training import predict

        if not hasattr(self, "fitted"):
            raise RuntimeError("Fit the relflow model before predicting.")
        return predict(self.fitted, task, db, table)
