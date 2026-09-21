"""Nori-Rel: the shared DFS feature stack with the frozen public Nori 30M regressor.

The featurization is not ours. Nori-Rel runs on RelArena's shared Deep Feature
Synthesis path (`relarena_core.featurization.dfs.build_dfs_features`), which is
built on `fastdfs` / featuretools and follows the DFS harness TabPFN-Rel
established here: depth-2 aggregation over the foreign-key graph, target-history
augmentation, temporal-diff features, and the anchor columns. Much of that in turn
comes from RDBLearn (https://github.com/HKUSHXLab/rdblearn). Cache keys, leak-safe
history handling, and the split protocol are shared, so a feature cache warmed for
TabPFN-Rel or RDBLearn is reused here as-is and
`python -m relarena.featurization.warm_cache` is the warmer for all three.

What differs from TabPFN-Rel, beyond swapping the tabular foundation model:

* **No training and no tuning.** Nori regresses by in-context learning, so `fit`
  only assembles the labeled context and never updates the checkpoint. The search
  space is the single config `{"max_depth": 2}`.
* **Regression only.** `supported_task_types` is `{REGRESSION}`.
* **Median output.** Nori predicts a distribution; the reported point estimate is
  its median rather than a mean.
* **An explicit large-context policy instead of a lossy one.** When the context
  exceeds the memory budget the adapter takes a seeded random window of training
  rows and keeps a full-precision BF16 cache offloaded to the host; `_memory_policy`
  sets `allow_quantization=False` and `allow_subsample=False`, so nothing is
  silently dropped. The context window and every forward pass are bounded by row
  budgets, so a large test table cannot blow up a single decode.
* **Pinned, verified weights.** The checkpoint is fetched at a fixed Hugging Face
  revision and its SHA-256 is checked before use, so an upstream reupload fails the
  run instead of quietly changing the numbers.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from relbench.base import Database, EntityTask, Table, TaskType

from relarena_core.featurization.dfs import DFS_MAX_DEPTH, build_dfs_features
from relarena_core.model import RelArenaModel
from relarena_core.registry import register_model
from relarena_core.search_space import SearchSpace

NORI_MODEL: Final = "nori-30m"
CHECKPOINT_REVISION: Final = "63c9f7facf9fb32c37ce3fc2fba331d524696318"
CHECKPOINT_SHA256: Final = (
    "818433f8af12c1137b96d9ff47e109b4eef5818d4e52a9656b2e573dbf13b74d"
)

_DEPTH = 2
_CONTEXT_ELEMENTS = 3_000_000
_CONTEXT_ROW_CHUNK = 512
_MAX_CONTEXT_ROWS = 60_000
_MAX_FORWARD_ROWS = 64_000
_MAX_EFFECTIVE_FEATURES = 512
_CACHE_TRIGGER_MARGIN_ROWS = 2

NORI_REL_SPACE = SearchSpace(default_overrides={"max_depth": _DEPTH})


def _load_nori() -> tuple[Any, Any]:
    """Load the optional Nori dependency only when the model is fitted."""
    from synthefy_nori import NoriRegressor, strict_pipeline

    return NoriRegressor, strict_pipeline


def _download_checkpoint() -> str:
    """Download the pinned public Nori 30M checkpoint."""
    from synthefy_nori.hf import download_checkpoint

    return download_checkpoint(model=NORI_MODEL, revision=CHECKPOINT_REVISION)


@lru_cache(maxsize=1)
def _sha256(path: Path, size: int, modified_ns: int) -> str:
    """Hash one checkpoint file version."""
    del size, modified_ns
    with path.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _checkpoint_path() -> str:
    """Return the pinned checkpoint after verifying its contents."""
    path = Path(_download_checkpoint())
    stat = path.stat()
    actual = _sha256(path, stat.st_size, stat.st_mtime_ns)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA mismatch for {path}: "
            f"expected {CHECKPOINT_SHA256}, got {actual}"
        )
    return str(path)


def _memory_policy() -> dict[str, float | bool | int]:
    """Use a host-offloaded BF16 cache without lossy fallbacks."""
    return {
        "allow_quantization": False,
        "allow_subsample": False,
        "context_row_chunk": _CONTEXT_ROW_CHUNK,
        "elements_budget": _CONTEXT_ELEMENTS,
        "gpu_budget_absolute_gb": 0.0,
    }


def _random_window(problem: Any, rng: np.random.Generator) -> np.ndarray:
    """Choose a seeded context window and keep query decoding cache-safe."""
    pool_size = min(problem.window, _MAX_CONTEXT_ROWS)
    pool = rng.permutation(problem.n_train)[:pool_size]
    max_query_chunk = _MAX_FORWARD_ROWS - pool_size
    if problem.window > _MAX_CONTEXT_ROWS:
        problem.query_chunk = max_query_chunk
        return problem.predict(pool)

    query = np.arange(problem.n_test)
    if problem.n_test <= problem.window:
        query = np.resize(query, problem.window + _CACHE_TRIGGER_MARGIN_ROWS)
    problem.query_chunk = min(len(query), max_query_chunk)
    return problem.predict(pool, query_idx=query)[: problem.n_test]


@register_model(search_space=NORI_REL_SPACE)
class NoriRelModel(RelArenaModel):
    """Depth-2 DFS features followed by the frozen public Nori 30M regressor."""

    name = "nori-rel"
    supported_task_types = frozenset({TaskType.REGRESSION})

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
        """Build DFS features and store the labeled Nori context."""
        del val_table, time_limit
        self._depth = int(self.config.get("max_depth", _DEPTH))
        if self._depth != _DEPTH:
            raise ValueError(f"Nori-Rel requires max_depth={_DEPTH}")

        self._history_table = train_table if task.time_col is not None else None
        features, categorical = self._features(task, db, train_table)
        features = features.reset_index(drop=True)
        if features.shape[1] == 0:
            raise ValueError("DFS produced no features at depth 2")
        self._columns = list(features.columns)

        n_features = min(features.shape[1], _MAX_EFFECTIVE_FEATURES)
        needs_window = (len(features) + 1) * n_features > _CONTEXT_ELEMENTS
        regressor, self._strict_pipeline = _load_nori()
        self._model = regressor(
            model_path=_checkpoint_path(),
            categorical_columns=categorical,
            memory_policy=_memory_policy(),
            large_context_policy=_random_window,
            large_context_threshold=1 if needs_window else len(features) + 1,
            large_context_seed=seed,
        )
        with self._strict_pipeline():
            self._model.fit(
                features, train_table.df[task.target_col].reset_index(drop=True)
            )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Return one median regression prediction per query row."""
        features, _ = self._features(task, db, table)
        features = features.reset_index(drop=True).reindex(columns=self._columns)
        with self._strict_pipeline():
            prediction = self._model.predict(features, output_type="median")

        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        if len(prediction) != len(table.df):
            raise RuntimeError(
                f"Nori returned {len(prediction)} predictions for "
                f"{len(table.df)} query rows"
            )
        return prediction

    def _features(
        self, task: EntityTask, db: Database, table: Table
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build the shared leak-safe DFS table."""
        return build_dfs_features(
            task,
            db,
            table,
            depth=self._depth,
            # Build (and cache) the deepest shared matrix and slice to our depth,
            # exactly as RDBLearn does: `max_depth` is part of the DFS cache key, so
            # passing our own depth here would key a private `max-depth-2` entry and
            # miss every cache warmed at the default depth.
            max_depth=DFS_MAX_DEPTH,
            history_table=self._history_table,
            keep_anchor_columns=True,
            cache=self.cache,
            run_identity=self.run_identity,
        )
