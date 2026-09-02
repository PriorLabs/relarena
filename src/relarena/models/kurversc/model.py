"""`kurversc` — validation-guided relational signal compression.

KurveRSC owns its complete prediction procedure, so it implements RelArena's
system contract rather than presenting its internal GraphReduce candidates as
harness hyperparameters. One system instance receives both temporal protocol
splits and runs two arms in sequence:

* **inner** — search GraphReduce configurations on the train/validation split,
  including full-frame reranking, and freeze both the selected declarative
  configuration and the learned feature-operation plan;
* **outer** — replay that exact configuration and operation plan on the
  test-censored database, fit the downstream CatBoost learner on the official
  train/validation data, and predict the masked test rows.

The frozen operation plan is learned state: carrying only the declarative graph
configuration would allow uncertain automatic annotations to change between
selection and reporting. The native system lifecycle keeps that state directly
inside one run; no module-global phase handoff is needed.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from relbench.base import Database, EntityTask, Table

from relarena.dataset import InnerSplit, OuterSplit
from relarena.registry import register_system
from relarena.system import RelArenaSystem

logger = logging.getLogger(__name__)


# This is the fixed, report-facing KurveRSC recipe. Internal GraphReduce
# candidates are selected by KurveRSC itself; they are not a RelArena search
# space and `--n-trials` therefore does not alter this procedure.
KURVERSC_DEFAULTS: Final[dict[str, Any]] = {
    "full_training_frames": 1,
    "search_training_frames": 1,
    "sample_rows": 50_000,
    "screening_rows": 10_000,
    "confirmation_top_k": 8,
    "search_full_data": True,
    "rerank_top_k": 3,
    "rerank_cutoff_frames": 3,
    "rerank_stability_penalty": 0.25,
    "feature_family_max_columns": 4,
    "adaptive_depth_promotion": True,
    "capability_pruning": True,
    "search_max_features": 8_000,
    "infer_ts_periods": False,
    "auto_text_features": False,
    "auto_annotate_max_text_columns": None,
    "duckdb_memory_limit": "64GB",
    "duckdb_max_temp_directory_size": "128GB",
}


@dataclass(frozen=True)
class _SelectedGraph:
    """The exact relational representation selected on validation."""

    config: Any
    execution_plan: dict[str, Any]


@register_system
class KurveRSCSystem(RelArenaSystem):
    """GraphReduce configuration search plus a CatBoost downstream learner."""

    name = "kurversc"
    schema_depth = 3

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        """Select on the inner split, refit on the outer split, and predict."""
        if time_limit is not None:
            logger.warning(
                "kurversc does not honor time_limit (%.0fs); its internal "
                "multi-fidelity search owns the evaluation budget.",
                time_limit,
            )

        inner_result, _ = self._fit_arm(
            task,
            inner_split.db_state,
            inner_split.train_table,
            inner_split.eval_table,
            seed=seed,
            phase="inner",
        )
        selection = _SelectedGraph(
            config=inner_result.recommended_config,
            execution_plan=copy.deepcopy(inner_result.execution_plan),
        )

        outer_result, outer_problem = self._fit_arm(
            task,
            outer_split.db_state,
            outer_split.train_table,
            outer_split.val_table,
            seed=seed,
            phase="outer",
            selection=selection,
        )
        return self._predict(
            task,
            outer_split.eval_table,
            result=outer_result,
            problem=outer_problem,
        )

    def _fit_arm(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table,
        *,
        seed: int,
        phase: str,
        selection: _SelectedGraph | None = None,
    ) -> tuple[Any, Any]:
        """Run one KurveRSC arm and return its fitted result and data adapter."""
        try:
            import kurversc
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The RelArena KurveRSC system requires the `kurversc` extra"
            ) from exc

        if phase not in {"inner", "outer"}:
            raise ValueError(f"Unknown KurveRSC phase: {phase!r}")
        if phase == "outer" and selection is None:
            raise RuntimeError("KurveRSC reporting arm requires the inner selection")

        config = KURVERSC_DEFAULTS
        full_training_frames = int(config["full_training_frames"])
        search_training_frames = int(config["search_training_frames"])
        rerank_top_k = int(config["rerank_top_k"])
        rerank_cutoff_frames = int(config["rerank_cutoff_frames"])

        # The selection arm needs enough historical cutoffs for its internal
        # reranker. The reporting arm reuses the frozen graph and materializes
        # only the requested production frames.
        adapter_train_timestamps = full_training_frames
        if phase == "inner":
            adapter_train_timestamps = max(
                full_training_frames,
                search_training_frames,
                rerank_cutoff_frames if rerank_top_k > 0 else 1,
            )

        identity = self.run_identity
        problem = kurversc.relbench_problem_from_objects(
            task,
            db,
            train_table,
            val_table,
            dataset_name=identity.dataset if identity is not None else "relbench",
            task_name=identity.task if identity is not None else None,
            sample_rows=int(config["sample_rows"]),
            search_full_data=bool(config["search_full_data"]),
            search_training_frames=search_training_frames,
            max_train_timestamps=adapter_train_timestamps,
            schema_depth=self.schema_depth,
            random_state=seed,
        )
        fit_kwargs: dict[str, Any] = {
            **problem.fit_kwargs(),
            **config,
            "random_state": seed,
        }
        if selection is not None:
            fit_kwargs["preselected_config"] = selection.config
            fit_kwargs["preselected_execution_plan"] = copy.deepcopy(
                selection.execution_plan
            )

        result = kurversc.fit(**fit_kwargs)
        return result, problem

    def _predict(
        self,
        task: EntityTask,
        table: Table,
        *,
        result: Any,
        problem: Any,
    ) -> np.ndarray:
        """Replay the reporting arm's frozen plan on masked test rows."""
        import kurversc

        prediction_rows = kurversc.Labels(
            table.df.copy(),
            key=task.entity_col,
            timestamp=task.time_col,
        )
        output = kurversc.predict(
            result,
            parent_node=problem.parent_node,
            prediction_node=prediction_rows,
            tables=problem.tables,
            relationships=problem.relationships,
            duckdb_memory_limit=KURVERSC_DEFAULTS["duckdb_memory_limit"],
            duckdb_max_temp_directory_size=(
                KURVERSC_DEFAULTS["duckdb_max_temp_directory_size"]
            ),
            use_validation_model=False,
        )
        return output["prediction"].to_numpy(dtype=float)


__all__ = ["KURVERSC_DEFAULTS", "KurveRSCSystem"]
