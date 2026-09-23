"""Frozen Nori 30M over relational features, with the backend routed by task.

Lineage. The harness around this model is RelArena's and follows TabPFN-Rel:
tasks, splits, inner-validation selection, refitting, scoring, and the shared
`build_dfs_features` generator. That generator enumerates candidates with
Featuretools (BSD-3-Clause), and RelArena describes the DFS-then-tabular-model
recipe as an adaptation of RDBLearn (Apache-2.0); no RDBLearn source is copied.
`text.py` adapts part of the TabPFN-Rel feature code under Apache-2.0 and keeps
that notice in its header.

What differs, beyond the tabular model:

- The feature backend is chosen by task type. Classification uses the shared
  generator above, unchanged. Regression uses NoriDFS, a planner and executor
  written for this model under `feature/`: bounded descendant, parent and
  association-bridge paths, point-in-time aggregates over windows at 1x, 4x and
  16x the task horizon, and a fixed feature budget. It does not import
  Featuretools or the shared generator.
- Regression adds target-side rules derived from the training labels alone:
  strict-past label lags, an integer lattice, a log scale for extremely skewed
  targets, a zero floor when the labels show one, and a recency-weighted
  training reservoir above the context cap. Each rule fails closed, and all of
  them are inert on a binary label, which is why classification does not use
  NoriDFS.
- Prose anchor columns are embedded with MiniLM and reduced to 16 SVD
  components fitted on the training split.
- Inference is exact: a random shared context under a 3M-element budget, no
  quantization and no context subsampling, with the memory rung Nori used
  audited after every prediction and any lossy fallback rejected.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table, TaskType

from relarena_core.featurization import build_dfs_features
from relarena_core.featurization.dfs import DFS_MAX_DEPTH
from relarena_core.model import RelArenaModel
from relarena_core.registry import register_model
from relarena_core.search_space import SearchSpace

from .context import (
    MAX_CONTEXT_ROWS,
    RESERVOIR_TEMPORAL_MIX,
    cache_safe_random_window,
    random_context_indices,
    training_reservoir,
)
from .history import StrictPastHistory
from .memory import (
    CONTEXT_ELEMENTS_BUDGET,
    NORIDFS_MEMORY_POLICY,
    validate_memory_report,
)
from .native import RelBenchNativeFeaturizer
from .profile import TargetProfile
from .recipe import (
    CONTEXT_PROFILE_V1,
    DENSE_HISTORY_ROWS_PER_ENTITY,
    TRAIN_PROFILE_V1,
    NoriDFSRecipe,
)
from .targets import TargetRules
from .text import anchor_text_columns, attach_anchor_text, pinned_minilm_encoder

logger = logging.getLogger(__name__)

GPU_BUDGET_GB_ENV: Final = "NORI_REL_GPU_BUDGET_GB"
HOST_BUDGET_GB_ENV: Final = "NORI_REL_HOST_BUDGET_GB"
DEFAULT_GPU_BUDGET_GB: Final = 0.0
#: Context rows per prefill chunk. `None` is unchunked, which is the memory
#: policy's own default and the value it falls back to chunking *from* after
#: an out-of-memory error (`FIT_ROW_CHUNK_ON_OOM`), so pinning a number also
#: spends that escalation in advance. Pinning 512 was measured to cost 10-14%
#: and buy nothing: study-adverse 678s -> 599s and item-sales 2,474s ->
#: 2,175s, both reproducing their scores to every digit.
CONTEXT_ROW_CHUNK: Final = None
MAX_EFFECTIVE_FEATURES: Final = 512
TEXT_SVD_DIM: Final = 16
TEXT_EMBEDDER: Final = "minilm"
FASTDFS_BACKEND: Final = "fastdfs"
NORIDFS_BACKEND: Final = "noridfs"
DEFAULT_CONFIG: Final = {
    "max_depth": 2,
    "context_policy": "random",
    "text_svd_dim": TEXT_SVD_DIM,
    "with_text": True,
}
#: One fixed NoriDFS configuration with every prediction heuristic off: 48
#: relational columns, one direct entity column, depth-2 paths, and windows at
#: 1x, 4x, and 16x the task horizon. Training labels enter as a strict-past
#: history table. This is the configuration compared against FastDFS.
NORIDFS_FIXED_CONFIG: Final = {
    **DEFAULT_CONFIG,
    "feature_backend": NORIDFS_BACKEND,
    "noridfs_max_paths": 24,
    "noridfs_max_features": 48,
    "noridfs_max_direct_features": 1,
    "noridfs_max_categorical_cardinality": 32,
    "noridfs_windows": "1,4,16",
    "noridfs_target_history": True,
}
#: The prediction heuristics shared by the tuned arms. Each one is derived from
#: the training split alone: lag features from past labels, a log scale for
#: extremely skewed targets, a zero floor when the labels show one, and a
#: recency-weighted 60k-row training reservoir above the context cap.
NORIDFS_HEURISTICS: Final = {
    "train_profile": TRAIN_PROFILE_V1,
    "noridfs_dense_history": True,
    "noridfs_target_lattice": True,
    "noridfs_skew_target_transform": True,
    "noridfs_nonnegative_support": True,
    "noridfs_train_reservoir": RESERVOIR_TEMPORAL_MIX,
    "nori_memory_policy": NORIDFS_MEMORY_POLICY,
}
NORIDFS_LEAN_CONFIG: Final = {**NORIDFS_FIXED_CONFIG, **NORIDFS_HEURISTICS}
NORIDFS_DIRECT_CONFIG: Final = {
    **NORIDFS_LEAN_CONFIG,
    "noridfs_max_features": 87,
    "noridfs_max_direct_features": 16,
    "noridfs_max_direct_categorical_cardinality": 32,
}
NORIDFS_MISSING_CONFIG: Final = {
    **NORIDFS_LEAN_CONFIG,
    "noridfs_max_features": 87,
    "noridfs_context_profile": CONTEXT_PROFILE_V1,
    "noridfs_missing_category_modes": True,
    "noridfs_missing_fractions": True,
}
#: RelArena scores each arm on the inner validation split and refits the
#: winner. The grid carries no dataset or task names.
NORIDFS_GRID: Final = [
    NORIDFS_LEAN_CONFIG,
    NORIDFS_DIRECT_CONFIG,
    NORIDFS_MISSING_CONFIG,
]
NORI_REL_SPACE: Final = SearchSpace(default_overrides=DEFAULT_CONFIG)
NORIDFS_SPACE: Final = SearchSpace(
    default_overrides=NORIDFS_LEAN_CONFIG, fixed_grid=NORIDFS_GRID
)
#: The registered entry routes on task type. Regression takes the NoriDFS grid;
#: classification takes RelArena's shared DFS generator, because every NoriDFS
#: prediction heuristic fails closed on a binary label and what remains is a
#: narrower feature set than the shared generator keeps. The key rides on every
#: arm so each recorded config states the rule rather than implying NoriDFS ran.
CLASSIFICATION_BACKEND: Final = FASTDFS_BACKEND
ROUTED_GRID: Final = [
    {**config, "classification_backend": CLASSIFICATION_BACKEND}
    for config in NORIDFS_GRID
]
ROUTED_SPACE: Final = SearchSpace(
    default_overrides=ROUTED_GRID[0], fixed_grid=ROUTED_GRID
)
NORI_MODEL: Final = "nori-30m"
CHECKPOINT_REVISION: Final = "63c9f7facf9fb32c37ce3fc2fba331d524696318"
CHECKPOINT_SHA256: Final = (
    "818433f8af12c1137b96d9ff47e109b4eef5818d4e52a9656b2e573dbf13b74d"
)


def _text_context_indices(n_rows: int, seed: int) -> np.ndarray:
    """Bound text preprocessing to the same seeded 60k context cap."""
    return random_context_indices(n_rows, seed, cap=MAX_CONTEXT_ROWS)


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


def _memory_policy() -> dict[str, float | bool | int | None]:
    """Build the exact no-subsampling inference policy."""
    policy: dict[str, float | bool | int | None] = {
        "allow_quantization": False,
        "allow_subsample": False,
        "context_row_chunk": CONTEXT_ROW_CHUNK,
        "elements_budget": CONTEXT_ELEMENTS_BUDGET,
    }
    gpu_budget = float(os.environ.get(GPU_BUDGET_GB_ENV, DEFAULT_GPU_BUDGET_GB))
    if gpu_budget < 0:
        raise ValueError(f"{GPU_BUDGET_GB_ENV} must be non-negative")
    policy["gpu_budget_absolute_gb"] = gpu_budget
    return policy


def _with_placement(policy: dict[str, Any]) -> dict[str, Any]:
    """Apply the runner's measured cache placement to a recipe's own policy.

    A recipe that carries `nori_memory_policy` used to win outright, so the
    per-task GPU budget the runner resolved was silently dropped and every
    task ran host-offloaded. Placement is the runner's call; everything that
    decides whether a run is bit-exact stays the recipe's.
    """
    placed = dict(policy)
    gpu_budget = os.environ.get(GPU_BUDGET_GB_ENV)
    if gpu_budget is not None:
        budget = float(gpu_budget)
        if budget < 0:
            raise ValueError(f"{GPU_BUDGET_GB_ENV} must be non-negative")
        placed["gpu_budget_absolute_gb"] = budget
        placed["offload_to_host"] = budget <= 0
    host_budget = os.environ.get(HOST_BUDGET_GB_ENV)
    if host_budget is not None:
        placed["host_budget_absolute_gb"] = float(host_budget)
    return placed


def _median_rows_per_entity(task: EntityTask, table: Table) -> float | None:
    """Measure repeated training cutoffs without reading target values."""
    entity_column = getattr(task, "entity_col", None)
    if not isinstance(entity_column, str) or entity_column not in table.df:
        return None
    counts = table.df[entity_column].value_counts(dropna=True, sort=False)
    counts = counts[counts > 0]
    return None if counts.empty else float(counts.median())


def _context_profile_active(
    recipe: NoriDFSRecipe, train_rows: int, median_rows_per_entity: float | None
) -> bool:
    """Identify large, repeated-entity contexts from training shape alone."""
    return bool(
        recipe.context_profile == CONTEXT_PROFILE_V1
        and train_rows > MAX_CONTEXT_ROWS
        and median_rows_per_entity is not None
        and 1 < median_rows_per_entity < DENSE_HISTORY_ROWS_PER_ENTITY
    )


def _context_feature_budget(configured: int, *, n_lags: int, text_width: int) -> int:
    """Reserve room for lag, age, and text features at the 60k-row cap."""
    total_width = CONTEXT_ELEMENTS_BUDGET // MAX_CONTEXT_ROWS
    return max(1, min(configured, total_width - 2 * n_lags - text_width))


def _noridfs_featurizer(
    recipe: NoriDFSRecipe,
    depth: int,
    *,
    context_profile: bool,
    n_lags: int,
    text_width: int,
) -> RelBenchNativeFeaturizer:
    """Build the NoriDFS adapter from the recipe and the resolved context shape."""
    from .feature.dfs.noridfs import NativeFeatureConfig

    max_features = recipe.max_features
    max_direct_features = recipe.max_direct_features
    missing_modes = recipe.missing_category_modes
    missing_fractions = recipe.missing_fractions
    if context_profile:
        max_features = _context_feature_budget(
            max_features, n_lags=n_lags, text_width=text_width
        )
        max_direct_features = 1
        missing_modes = False
        missing_fractions = False
    return RelBenchNativeFeaturizer(
        NativeFeatureConfig(
            max_depth=depth,
            max_paths=recipe.max_paths,
            max_features=max_features,
            max_direct_features=max_direct_features,
            max_direct_categorical_cardinality=recipe.max_direct_categorical_cardinality,
            max_categorical_cardinality=recipe.max_categorical_cardinality,
            window_multipliers=recipe.windows,
            include_missing_in_modes=missing_modes,
            include_missing_fractions=missing_fractions,
        ),
        with_target_history=recipe.target_history,
    )


def _subset_table(table: Table, rows: np.ndarray) -> Table:
    """Return a RelBench table holding only the selected rows, in order."""
    return Table(
        df=table.df.iloc[rows].reset_index(drop=True),
        fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


class NoriRelModel(RelArenaModel):
    """RelArena relational features followed by frozen Nori inference."""

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
        depth = int(self.config.get("max_depth", 2))
        context_policy = self.config.get("context_policy", "random")
        text_svd_dim = int(self.config.get("text_svd_dim", TEXT_SVD_DIM))
        with_text = bool(self.config.get("with_text", False))
        if (
            depth != 2
            or context_policy != "random"
            or text_svd_dim != TEXT_SVD_DIM
            or not with_text
        ):
            raise ValueError(
                "the Nori-Rel text experiment requires depth 2, random context, "
                f"{TEXT_SVD_DIM} text SVD features, and text"
            )
        backend = str(self.config.get("feature_backend", FASTDFS_BACKEND))
        if backend not in {FASTDFS_BACKEND, NORIDFS_BACKEND}:
            raise ValueError(f"unknown feature_backend {backend!r}")
        if backend == FASTDFS_BACKEND and any(
            key == "train_profile" or key.startswith("noridfs_") for key in self.config
        ):
            raise ValueError(
                "noridfs_* and train_profile keys need the noridfs backend"
            )

        self._depth = depth
        self._backend = backend
        self._history_table = train_table if task.time_col is not None else None
        self._anchor_text_columns = anchor_text_columns(db, task)
        self._noridfs = None
        self._history = None
        self._rules = TargetRules()
        self._decoder = "median"
        self._memory_request: dict[str, Any] | None = None
        target = train_table.df[task.target_col].reset_index(drop=True)
        fit_table = train_table
        if backend == NORIDFS_BACKEND:
            fit_table, target = self._fit_noridfs(task, db, train_table, target, seed)

        features, categorical = self._features(task, db, fit_table)
        features = features.reset_index(drop=True)
        if self._history is not None:
            features = self._history.transform(features, task, fit_table)
        if self._anchor_text_columns:
            context = _text_context_indices(len(features), seed)
            features = features.iloc[context].reset_index(drop=True)
            target = target.iloc[context].reset_index(drop=True)
            split = fit_table.df.iloc[context].reset_index(drop=True)
            features, self._text_columns = attach_anchor_text(
                features,
                db,
                task,
                split,
                self._anchor_text_columns,
                strict_cutoff=backend == NORIDFS_BACKEND,
            )
        else:
            self._text_columns = []
        # A column can be entirely missing in the training context, after the
        # strict temporal cutoff or the text row cap, while still carrying values
        # at predict. Drop it and freeze the set, so a prediction-only value never
        # reaches a column Nori fitted as empty.
        features = features.dropna(axis=1, how="all")
        if features.shape[1] == 0:
            raise ValueError(
                "no features remain after dropping all-missing training columns"
            )
        kept = set(features.columns)
        categorical = [column for column in categorical if column in kept]
        self._text_columns = [column for column in self._text_columns if column in kept]
        self._columns = list(features.columns)

        text_width = text_svd_dim if self._text_columns else 0
        effective_features = min(
            features.shape[1] - len(self._text_columns) + text_width,
            MAX_EFFECTIVE_FEATURES,
        )
        needs_context_window = (
            len(features) + 1
        ) * effective_features > CONTEXT_ELEMENTS_BUDGET
        model_options: dict[str, Any] = {
            "model_path": _checkpoint_path(),
            "categorical_columns": categorical,
            "memory_policy": self._memory_request or _memory_policy(),
            "large_context_policy": (
                cache_safe_random_window if backend == NORIDFS_BACKEND else "random"
            ),
            "large_context_threshold": 1 if needs_context_window else len(features) + 1,
            "large_context_seed": seed,
        }
        if self._rules.discretization is not None:
            model_options["discretize"] = self._rules.discretization
            model_options["categorical_levels"] = self._rules.lattice
        if self._text_columns:
            model_options.update(
                text_columns=self._text_columns,
                svd_dim=text_svd_dim,
                embedder=(
                    pinned_minilm_encoder
                    if backend == NORIDFS_BACKEND
                    else TEXT_EMBEDDER
                ),
            )
        regressor, self._strict_pipeline = _load_nori()
        self._model = regressor(**model_options)
        with self._strict_pipeline():
            self._model.fit(features, self._rules.forward(target))

    def _fit_noridfs(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        target: pd.Series,
        seed: int,
    ) -> tuple[Table, pd.Series]:
        """Resolve the recipe, bound the training rows, and fit the featurizer."""
        recipe = NoriDFSRecipe.from_config(self.config)
        if recipe.memory_policy is not None:
            # Placed once, so the audit checks the policy the run actually used.
            self._memory_request = _with_placement(dict(recipe.memory_policy))
        median_rows = _median_rows_per_entity(task, train_table)
        profile = (
            TargetProfile.fit(target, task.task_type)
            if recipe.uses_history_lags
            else TargetProfile()
        )
        # A bounded target that mostly sits on its endpoints, such as a success
        # rate, is not helped by its own lags; that route keeps the median.
        n_lags = recipe.history_lags(median_rows)
        if task.time_col is None or profile.endpoint_rich_bounded:
            n_lags = 0
        elif profile.low_skew_many_level:
            self._decoder = "mean"
        self._rules = TargetRules.fit(
            target,
            task.task_type,
            lattice=recipe.target_lattice,
            skew_transform=recipe.skew_target_transform,
            nonnegative=recipe.nonnegative_support,
        )
        if self._rules.discretization is not None:
            self._decoder = "mean"

        fit_table = train_table
        if len(train_table.df) > MAX_CONTEXT_ROWS:
            rows = training_reservoir(task, train_table, seed, recipe.reservoir)
            fit_table = _subset_table(train_table, rows)
            target = target.iloc[rows].reset_index(drop=True)
        self._noridfs = _noridfs_featurizer(
            recipe,
            self._depth,
            context_profile=_context_profile_active(
                recipe, len(train_table.df), median_rows
            ),
            n_lags=n_lags,
            text_width=TEXT_SVD_DIM if self._anchor_text_columns else 0,
        ).fit(task, db, fit_table, history_table=self._history_table)
        if n_lags:
            # The lag pool is the full authorized training split, not the
            # reservoir, so every query row sees its complete strict past.
            self._history = StrictPastHistory(n_lags).fit(task, train_table)
        return fit_table, target

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Return one task-appropriate prediction per query row."""
        features, _ = self._features(task, db, table)
        if self._history is not None:
            features = self._history.transform(features, task, table)
        features, _ = attach_anchor_text(
            features,
            db,
            task,
            table.df,
            self._anchor_text_columns,
            strict_cutoff=self._backend == NORIDFS_BACKEND,
        )
        features = features.reset_index(drop=True).reindex(columns=self._columns)
        binary = task.task_type == TaskType.BINARY_CLASSIFICATION
        with self._strict_pipeline():
            prediction = self._model.predict(
                features,
                output_type="mean" if binary else self._decoder,
            )
        if self._memory_request is not None:
            self.memory_audit_ = validate_memory_report(
                getattr(self._model, "memory_report_", None), self._memory_request
            )
            logger.info("nori inference memory rung: %s", self.memory_audit_)

        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        prediction = self._rules.inverse(prediction)
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
        """Materialize the configured relational feature table."""
        if self._noridfs is not None:
            return self._noridfs.transform(task, db, table)
        return build_dfs_features(
            task,
            db,
            table,
            depth=self._depth,
            # The cache key follows `max_depth`. Build the shared deepest matrix
            # and slice to our depth, as RDBLearn does, so a cache warmed at the
            # default depth hits. The slice matches a direct depth-2 build after
            # Nori's float32 cast; the float64 values differ only by rounding.
            max_depth=DFS_MAX_DEPTH,
            history_table=self._history_table,
            keep_anchor_columns=True,
            cache=self.cache,
            run_identity=self.run_identity,
        )


@register_model(search_space=ROUTED_SPACE)
class NoriRel(NoriRelModel):
    """The public 30M checkpoint, with the feature backend chosen by task type.

    Regression runs the three NoriDFS arms and refits the winner. Classification
    runs RelArena's shared DFS generator at its single default configuration,
    whichever arm it was handed: the arms differ only in NoriDFS keys, which have
    nothing to act on there, so every arm fits the same model.
    """

    name = "nori-rel"

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
        """Resolve the route, then fit the backend it names."""
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            self.config = dict(DEFAULT_CONFIG)
        super().fit(task, db, train_table, val_table, seed=seed, time_limit=time_limit)


class NoriRelNoriDFS(NoriRelModel):
    """Unregistered regression candidate: NoriDFS features, same frozen Nori."""

    name = "nori-rel-noridfs"
    supported_task_types = frozenset({TaskType.REGRESSION})

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Default to the lean tuned arm when no full configuration is given."""
        super().__init__(
            dict(NORIDFS_LEAN_CONFIG) if config is None else config, **kwargs
        )
