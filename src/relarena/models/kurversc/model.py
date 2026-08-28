"""KurveRSC system integration.

KurveRSC performs its own validation-guided GraphReduce configuration search,
so this is a RelArena *system* with a parameter-free harness search space. The
inner phase selects the graph configuration on train/validation. The outer
phase reuses that exact configuration, learns a production operation plan on
the full official training rows, refits CatBoost on train+validation, and
replays the frozen plan for test prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table

from relarena.identity import RunIdentity
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace

KURVERSC_SPACE = SearchSpace(
    default_overrides={
        "full_training_frames": 1,
        "sample_rows": 100_000,
        "feature_family_max_columns": 4,
    }
)

# RelArena's current experimental system contract creates a fresh object for
# the outer phase. Keep only the selected declarative graph configuration,
# keyed by the stable run identity; fitted estimators and data never cross the
# inner/outer boundary.
_SELECTED_CONFIGS: dict[tuple[str, str | None, int], Any] = {}


def _selection_key(
    identity: RunIdentity | None, seed: int
) -> tuple[str, str | None, int]:
    if identity is None:
        return ("unidentified", None, seed)
    return (identity.dataset, identity.task, seed)


def _table_with_frame(table: Table, frame: pd.DataFrame) -> Table:
    return Table(
        df=frame.reset_index(drop=True),
        fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def _recover_outer_validation(task: EntityTask, combined: Table) -> tuple[Table, Table]:
    """Recover official train/val parts from RelArena's refit union."""
    dataset = getattr(task, "dataset", None)
    boundary = getattr(dataset, "val_timestamp", None)
    if boundary is None:
        raise ValueError(
            "KurveRSC outer refit needs task.dataset.val_timestamp to recover "
            "the official train/validation boundary"
        )
    timestamps = pd.to_datetime(combined.df[task.time_col])
    is_validation = timestamps >= pd.Timestamp(boundary)
    if not is_validation.any() or is_validation.all():
        raise ValueError(
            "The outer train+validation table did not contain both sides of "
            "task.dataset.val_timestamp"
        )
    return (
        _table_with_frame(combined, combined.df.loc[~is_validation]),
        _table_with_frame(combined, combined.df.loc[is_validation]),
    )


@register_model(search_space=KURVERSC_SPACE)
class KurveRSCSystem(RelArenaModel):
    """GraphReduce configuration search plus a CatBoost downstream model."""

    name = "kurversc"
    kind = "system"
    refit_on_full_data = True

    sample_rows = 100_000
    search_training_frames = 1
    schema_depth = 3

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
        """Run KurveRSC within RelArena's censored phase database."""
        try:
            import kurversc
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The RelArena KurveRSC model requires the `kurversc` extra"
            ) from exc

        unknown = set(self.config) - {
            "full_training_frames",
            "sample_rows",
            "feature_family_max_columns",
        }
        if unknown:
            raise ValueError(f"Unknown KurveRSC configuration keys: {sorted(unknown)}")
        full_training_frames = int(self.config.get("full_training_frames", 1))
        if full_training_frames < 1:
            raise ValueError("full_training_frames must be positive")
        sample_rows = int(self.config.get("sample_rows", self.sample_rows))
        if sample_rows < 2:
            raise ValueError("sample_rows must be at least 2")
        raw_family_cap = self.config.get("feature_family_max_columns", 4)
        feature_family_max_columns = (
            None if raw_family_cap is None else int(raw_family_cap)
        )
        if feature_family_max_columns is not None and feature_family_max_columns < 1:
            raise ValueError("feature_family_max_columns must be positive or null")

        phase = self.run_identity.phase if self.run_identity is not None else None
        if val_table is None:
            train_table, val_table = _recover_outer_validation(task, train_table)
        key = _selection_key(self.run_identity, seed)
        selected = _SELECTED_CONFIGS.get(key) if phase == "outer" else None

        problem = kurversc.relbench_problem_from_objects(
            task,
            db,
            train_table,
            val_table,
            dataset_name=(
                self.run_identity.dataset
                if self.run_identity is not None
                else "relbench"
            ),
            task_name=(
                self.run_identity.task if self.run_identity is not None else None
            ),
            sample_rows=sample_rows,
            max_train_timestamps=full_training_frames,
            schema_depth=self.schema_depth,
            random_state=seed,
        )
        fit_kwargs: dict[str, Any] = {
            **problem.fit_kwargs(),
            "sample_rows": sample_rows,
            "feature_family_max_columns": feature_family_max_columns,
            "search_training_frames": self.search_training_frames,
            "full_training_frames": full_training_frames,
            "random_state": seed,
        }
        if selected is not None:
            fit_kwargs["graph_configs"] = (selected,)
        self._result = kurversc.fit(**fit_kwargs)
        self._problem = problem
        self._use_validation_model = phase != "outer"
        if phase != "outer":
            _SELECTED_CONFIGS[key] = self._result.best_config

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Replay the fitted operation plan and return contract predictions."""
        import kurversc

        prediction_rows = kurversc.Labels(
            table.df.copy(),
            key=task.entity_col,
            timestamp=task.time_col,
        )
        output = kurversc.predict(
            self._result,
            parent_node=self._problem.parent_node,
            prediction_node=prediction_rows,
            tables=self._problem.tables,
            relationships=self._problem.relationships,
            use_validation_model=self._use_validation_model,
        )
        return output["prediction"].to_numpy(dtype=float)
