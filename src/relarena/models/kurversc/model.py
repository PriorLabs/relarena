"""KurveRSC system integration.

KurveRSC performs its own validation-guided GraphReduce configuration search,
so this is a RelArena *system* with a parameter-free harness search space. The
inner phase selects the graph configuration on train/validation. The outer
phase reuses that exact configuration and frozen GraphReduce operation plan,
refits CatBoost on train+validation, and replays the plan for test prediction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
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
        "search_training_frames": 1,
        "sample_rows": 50_000,
        "screening_rows": 10_000,
        "confirmation_top_k": 8,
        "rerank_top_k": 3,
        "rerank_cutoff_frames": 3,
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
)


@dataclass(frozen=True)
class _SelectedGraph:
    """KurveRSC selection state that must survive RelArena's phase boundary."""

    config: Any
    execution_plan: dict[str, Any]


# RelArena creates a fresh model object for the outer phase. The GraphReduce
# operation plan is learned state, not merely an implementation detail, so it
# travels with the selected declarative configuration. Estimators and data do
# not cross the inner/outer boundary.
_SELECTED_GRAPHS: dict[tuple[str, str | None, int], _SelectedGraph] = {}


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
            "search_training_frames",
            "sample_rows",
            "screening_rows",
            "confirmation_top_k",
            "search_full_data",
            "rerank_top_k",
            "rerank_cutoff_frames",
            "rerank_stability_penalty",
            "feature_family_max_columns",
            "adaptive_depth_promotion",
            "capability_pruning",
            "search_max_features",
            "infer_ts_periods",
            "auto_text_features",
            "auto_annotate_max_text_columns",
            "duckdb_memory_limit",
            "duckdb_max_temp_directory_size",
        }
        if unknown:
            raise ValueError(f"Unknown KurveRSC configuration keys: {sorted(unknown)}")
        full_training_frames = int(self.config.get("full_training_frames", 1))
        if full_training_frames < 1:
            raise ValueError("full_training_frames must be positive")
        search_training_frames = int(
            self.config.get("search_training_frames", self.search_training_frames)
        )
        if search_training_frames < 1:
            raise ValueError("search_training_frames must be positive")
        sample_rows = int(self.config.get("sample_rows", self.sample_rows))
        if sample_rows < 2:
            raise ValueError("sample_rows must be at least 2")
        screening_rows = int(self.config.get("screening_rows", 10_000))
        if screening_rows < 1:
            raise ValueError("screening_rows must be positive")
        confirmation_top_k = int(self.config.get("confirmation_top_k", 8))
        if confirmation_top_k < 0:
            raise ValueError("confirmation_top_k must be non-negative")
        search_full_data = bool(self.config.get("search_full_data", False))
        rerank_top_k = int(self.config.get("rerank_top_k", 3))
        if rerank_top_k < 0:
            raise ValueError("rerank_top_k must be non-negative")
        rerank_cutoff_frames = int(self.config.get("rerank_cutoff_frames", 3))
        if rerank_cutoff_frames < 1:
            raise ValueError("rerank_cutoff_frames must be positive")
        rerank_stability_penalty = float(
            self.config.get("rerank_stability_penalty", 0.25)
        )
        if rerank_stability_penalty < 0:
            raise ValueError("rerank_stability_penalty must be non-negative")
        raw_family_cap = self.config.get("feature_family_max_columns", 4)
        feature_family_max_columns = (
            None if raw_family_cap is None else int(raw_family_cap)
        )
        if feature_family_max_columns is not None and feature_family_max_columns < 1:
            raise ValueError("feature_family_max_columns must be positive or null")
        adaptive_depth_promotion = bool(
            self.config.get("adaptive_depth_promotion", True)
        )
        capability_pruning = bool(self.config.get("capability_pruning", True))
        raw_search_max_features = self.config.get("search_max_features", 8_000)
        search_max_features = (
            None if raw_search_max_features is None else int(raw_search_max_features)
        )
        if search_max_features is not None and search_max_features < 1:
            raise ValueError("search_max_features must be positive or null")
        infer_ts_periods = bool(self.config.get("infer_ts_periods", False))
        auto_text_features = bool(self.config.get("auto_text_features", False))
        raw_text_cap = self.config.get("auto_annotate_max_text_columns")
        auto_annotate_max_text_columns = (
            None if raw_text_cap is None else int(raw_text_cap)
        )
        if (
            auto_annotate_max_text_columns is not None
            and auto_annotate_max_text_columns < 1
        ):
            raise ValueError("auto_annotate_max_text_columns must be positive or null")
        duckdb_memory_limit = self.config.get("duckdb_memory_limit", "64GB")
        if not isinstance(duckdb_memory_limit, str) or not duckdb_memory_limit.strip():
            raise ValueError("duckdb_memory_limit must be a non-empty string")
        duckdb_max_temp_directory_size = self.config.get(
            "duckdb_max_temp_directory_size", "128GB"
        )
        if (
            not isinstance(duckdb_max_temp_directory_size, str)
            or not duckdb_max_temp_directory_size.strip()
        ):
            raise ValueError(
                "duckdb_max_temp_directory_size must be a non-empty string"
            )

        phase = self.run_identity.phase if self.run_identity is not None else None
        if val_table is None:
            train_table, val_table = _recover_outer_validation(task, train_table)
        key = _selection_key(self.run_identity, seed)
        selected = _SELECTED_GRAPHS.get(key) if phase == "outer" else None

        # Final fitting and temporal configuration reranking have independent
        # frame budgets.  Keep enough historical cutoffs for the reranker in
        # the inner phase even when production training intentionally uses
        # only the latest frame.  The outer phase reuses the selected graph
        # and therefore needs only the requested production frames.
        adapter_train_timestamps = full_training_frames
        if selected is None:
            adapter_train_timestamps = max(
                full_training_frames,
                search_training_frames,
                rerank_cutoff_frames if rerank_top_k > 0 else 1,
            )

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
            search_full_data=search_full_data,
            search_training_frames=search_training_frames,
            max_train_timestamps=adapter_train_timestamps,
            schema_depth=self.schema_depth,
            random_state=seed,
        )
        fit_kwargs: dict[str, Any] = {
            **problem.fit_kwargs(),
            "sample_rows": sample_rows,
            "screening_rows": screening_rows,
            "confirmation_top_k": confirmation_top_k,
            "search_full_data": search_full_data,
            "rerank_top_k": rerank_top_k,
            "rerank_cutoff_frames": rerank_cutoff_frames,
            "rerank_stability_penalty": rerank_stability_penalty,
            "feature_family_max_columns": feature_family_max_columns,
            "adaptive_depth_promotion": adaptive_depth_promotion,
            "capability_pruning": capability_pruning,
            "search_max_features": search_max_features,
            "infer_ts_periods": infer_ts_periods,
            "auto_text_features": auto_text_features,
            "auto_annotate_max_text_columns": auto_annotate_max_text_columns,
            "duckdb_memory_limit": duckdb_memory_limit,
            "duckdb_max_temp_directory_size": duckdb_max_temp_directory_size,
            "search_training_frames": search_training_frames,
            "full_training_frames": full_training_frames,
            "random_state": seed,
        }
        if selected is not None:
            fit_kwargs["preselected_config"] = selected.config
            fit_kwargs["preselected_execution_plan"] = copy.deepcopy(
                selected.execution_plan
            )
        self._result = kurversc.fit(**fit_kwargs)
        self._problem = problem
        self._use_validation_model = phase != "outer"
        self._duckdb_memory_limit = duckdb_memory_limit
        self._duckdb_max_temp_directory_size = duckdb_max_temp_directory_size
        if phase != "outer":
            _SELECTED_GRAPHS[key] = _SelectedGraph(
                config=self._result.recommended_config,
                execution_plan=copy.deepcopy(self._result.execution_plan),
            )

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
            duckdb_memory_limit=self._duckdb_memory_limit,
            duckdb_max_temp_directory_size=(self._duckdb_max_temp_directory_size),
            use_validation_model=self._use_validation_model,
        )
        return output["prediction"].to_numpy(dtype=float)
