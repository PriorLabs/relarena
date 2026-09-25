"""Frozen Nori 30M over RelArena's relational features.

Lineage. The harness around this model is RelArena's and follows TabPFN-Rel:
tasks, splits, refitting, scoring, and the shared `build_dfs_features`
generator, which enumerates candidates with Featuretools (BSD-3-Clause).
RelArena describes this DFS-then-tabular-model recipe as an adaptation of
RDBLearn (Apache-2.0); no RDBLearn source is copied. `text.py` adapts part of
the TabPFN-Rel feature code under Apache-2.0 and keeps that notice in its
header.

What differs, beyond the tabular model:

- The tabular model is the frozen public Nori 30M checkpoint, used in context.
  Nothing is trained: `fit` assembles the context and `predict` runs one
  forward pass.
- Prose anchor columns are embedded with MiniLM and reduced to 16 SVD
  components fitted on the training split.
- Inference is exact: a random context under a 3M-element budget, with no
  quantization and no context subsampling.

Every task uses the same features and the same single configuration.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table, TaskType

from relarena_core.featurization import build_dfs_features
from relarena_core.model import RelArenaModel
from relarena_core.registry import register_model
from relarena_core.search_space import SearchSpace

from .text import STRINGIFIED_NULLS, anchor_text_columns, attach_anchor_text

#: Relational depth of the features Nori sees.
DFS_DEPTH: Final = 2
#: Context rows kept for text preprocessing, drawn with the task seed.
MAX_TEXT_CONTEXT_ROWS: Final = 60_000
#: Rows times features above which Nori draws a random context window.
CONTEXT_ELEMENTS_BUDGET: Final = 3_000_000
MAX_EFFECTIVE_FEATURES: Final = 512
TEXT_SVD_DIM: Final = 16
TEXT_EMBEDDER: Final = "minilm"
#: GPU memory Nori may keep its context cache in; 0 keeps it in host memory.
#: Placement changes speed, not predictions.
GPU_BUDGET_GB_ENV: Final = "NORI_REL_GPU_BUDGET_GB"
NORI_MODEL: Final = "nori-30m"
CHECKPOINT_REVISION: Final = "63c9f7facf9fb32c37ce3fc2fba331d524696318"
CHECKPOINT_SHA256: Final = (
    "818433f8af12c1137b96d9ff47e109b4eef5818d4e52a9656b2e573dbf13b74d"
)
NORI_REL_SPACE: Final = SearchSpace(default_overrides={})


def _load_nori() -> tuple[Any, Any]:
    """Load optional Nori dependencies only when fitting."""
    from synthefy_nori import NoriRegressor, strict_pipeline

    return NoriRegressor, strict_pipeline


def _download_checkpoint() -> str:
    """Download the pinned public Nori checkpoint."""
    from synthefy_nori.hf import download_checkpoint

    return download_checkpoint(model=NORI_MODEL, revision=CHECKPOINT_REVISION)


@lru_cache(maxsize=1)
def _sha256(path: Path, size: int, modified_ns: int) -> str:
    """Hash a checkpoint once per file version."""
    del size, modified_ns
    with path.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _checkpoint_path() -> str:
    """Download and verify the exact public 30M checkpoint."""
    path = Path(_download_checkpoint())
    stat = path.stat()
    actual = _sha256(path, stat.st_size, stat.st_mtime_ns)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA mismatch for {path}: "
            f"expected {CHECKPOINT_SHA256}, got {actual}"
        )
    return str(path)


def _memory_policy() -> dict[str, Any]:
    """Exact inference: no quantization and no context subsampling."""
    gpu_budget = float(os.environ.get(GPU_BUDGET_GB_ENV, 0.0))
    if gpu_budget < 0:
        raise ValueError(f"{GPU_BUDGET_GB_ENV} must be non-negative")
    return {
        "allow_quantization": False,
        "allow_subsample": False,
        "context_row_chunk": None,
        "elements_budget": CONTEXT_ELEMENTS_BUDGET,
        "gpu_budget_absolute_gb": gpu_budget,
    }


def _text_context_rows(n_rows: int, seed: int) -> np.ndarray:
    """Every row up to the text cap, otherwise a seeded random subset."""
    if n_rows <= MAX_TEXT_CONTEXT_ROWS:
        return np.arange(n_rows)
    return np.random.default_rng(seed).permutation(n_rows)[:MAX_TEXT_CONTEXT_ROWS]


@register_model(search_space=NORI_REL_SPACE)
class NoriRel(RelArenaModel):
    """RelArena relational features followed by frozen Nori inference."""

    name = "nori-rel"
    supported_task_types = frozenset(
        {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION}
    )

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
        """Build relational features and store the Nori context."""
        del val_table, time_limit
        self._history_table = train_table if task.time_col is not None else None
        self._anchor_text_columns = anchor_text_columns(db, task)
        target = train_table.df[task.target_col].reset_index(drop=True)

        features, categorical = self._features(task, db, train_table)
        features = features.reset_index(drop=True)
        text_columns: list[str] = []
        if self._anchor_text_columns:
            rows = _text_context_rows(len(features), seed)
            features = features.iloc[rows].reset_index(drop=True)
            target = target.iloc[rows].reset_index(drop=True)
            split = train_table.df.iloc[rows].reset_index(drop=True)
            features, text_columns = attach_anchor_text(
                features, db, task, split, self._anchor_text_columns
            )

        # A column entirely missing in training can still carry values at
        # predict, so drop it and freeze the set. Text arrives with its nulls
        # already stringified, so an empty text column is a run of placeholders.
        empty_text = [
            column
            for column in text_columns
            if (
                features[column].isna() | features[column].isin(STRINGIFIED_NULLS)
            ).all()
        ]
        features = features.drop(columns=empty_text).dropna(axis=1, how="all")
        if features.shape[1] == 0:
            raise ValueError("no features remain after dropping empty columns")
        kept = set(features.columns)
        categorical = [column for column in categorical if column in kept]
        text_columns = [column for column in text_columns if column in kept]
        self._columns = list(features.columns)

        text_width = TEXT_SVD_DIM if text_columns else 0
        effective_features = min(
            features.shape[1] - len(text_columns) + text_width,
            MAX_EFFECTIVE_FEATURES,
        )
        needs_window = (
            len(features) + 1
        ) * effective_features > CONTEXT_ELEMENTS_BUDGET
        options: dict[str, Any] = {
            "model_path": _checkpoint_path(),
            "categorical_columns": categorical,
            "memory_policy": _memory_policy(),
            "large_context_policy": "random",
            "large_context_threshold": 1 if needs_window else len(features) + 1,
            "large_context_seed": seed,
        }
        if text_columns:
            options.update(
                text_columns=text_columns, svd_dim=TEXT_SVD_DIM, embedder=TEXT_EMBEDDER
            )
        regressor, self._strict_pipeline = _load_nori()
        self._model = regressor(**options)
        with self._strict_pipeline():
            self._model.fit(features, target)

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Return the median for regression and a clipped mean risk score."""
        features, _ = self._features(task, db, table)
        features, _ = attach_anchor_text(
            features, db, task, table.df, self._anchor_text_columns
        )
        features = features.reset_index(drop=True).reindex(columns=self._columns)
        binary = task.task_type == TaskType.BINARY_CLASSIFICATION
        with self._strict_pipeline():
            prediction = self._model.predict(
                features, output_type="mean" if binary else "median"
            )
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        if binary:
            # RelArena scores ROC-AUC, so keep the continuous risk score.
            prediction = np.clip(prediction, 0.0, 1.0)
        if len(prediction) != len(table.df):
            raise RuntimeError(
                f"Nori returned {len(prediction)} predictions for "
                f"{len(table.df)} query rows"
            )
        return prediction

    def _features(
        self, task: EntityTask, db: Database, table: Table
    ) -> tuple[pd.DataFrame, list[str]]:
        """Shared DFS features built directly at depth 2.

        Reading the shared depth-4 cache instead moved rel-event/user-attendance
        MAE from 0.247 to 0.283 in a benchmark run, though an uncached depth-4
        slice of the training table matches a direct build.
        """
        return build_dfs_features(
            task,
            db,
            table,
            depth=DFS_DEPTH,
            max_depth=DFS_DEPTH,
            history_table=self._history_table,
            keep_anchor_columns=True,
            cache=self.cache,
            run_identity=self.run_identity,
        )
