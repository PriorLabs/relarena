"""`tabpfn-rel` — DFS + a TabPFN foundation model, with opt-in feature/context extras.

Like the open `rdblearn` model, it solves a RelBench entity task by deep feature
synthesis (DFS) — aggregating the linked tables into one flat row per (entity, cutoff)
— and feeding that table to a TabPFN foundation model that predicts the test rows by
in-context learning over the train rows. On top of that it adds four opt-in knobs,
each recovering signal flat-DFS-into-one-TFM-call leaves behind (see
`relarena.models.tabpfn_rel.features` and
`relarena.models.tabpfn_rel.context`):

  1. **Calendar features** — cyclic sin/cos of the cutoff timestamp.
  2. **History lags** — per-entity lags of the target's own past values.
  3. **Text features** — anchor-table free text passed through raw, for a TFM whose
     estimator consumes text columns itself (the hosted-API backend).
  4. **Recency-aware contexts** — `soft_pool` / `hard_pool` build the TFM's
     in-context examples from a recency-weighted pool instead of a uniform sample.

`fit` is the whole story top to bottom: build DFS features for the full train labels
(cached on a warm run; see `relarena.featurization.cache`), apply the
enabled feature extras, then fit the TFM — a seeded downsample for the default
`random` context, or the recency pool for `soft_pool` / `hard_pool`. The expensive
DFS matrix is content-cached, so the downsample / pool selection happens cheaply
afterward (rather than being fused into a pre-DFS slice); text is handled inside
the estimator at fit time.

Two variants are registered, sharing the config validated in the reference sweeps —
TabPFN v3 with hard-pool recency contexts (`K=100k`, `M=4·K`), tuned over DFS depth
only: `tabpfn-rel-local` runs the local TabPFN v3, and `tabpfn-rel-client` runs
through the hosted TabPFN API with anchor-table text passed through raw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from relbench.base import Database, EntityTask, Table

from relarena.featurization import DFS_MAX_DEPTH, build_dfs_features
from relarena.model import RelArenaModel
from relarena.models._shared.tfm.tfm import TFM_REGISTRY, predict_tfm
from relarena.models.tabpfn_rel.context import ContextStrategy
from relarena.models.tabpfn_rel.features import FeaturePipeline
from relarena.registry import register_model
from relarena.search_space import SearchSpace

#: Depth grid lower bound. Shallower depths yield no DFS features for entity tasks.
_MIN_DEPTH = 2

#: The config validated in the reference sweeps: TabPFN v3 with hard-pool recency
#: contexts (K=100k per estimator, pool M=4·K). Only the DFS depth is tuned.
_DEFAULT_KNOBS: dict[str, Any] = {
    "tfm": "tabpfn-v3",
    "context_strategy": "hard_pool",
    "subsample_samples": 100_000,
    "pool_inflation": 4.0,
    "with_text": False,
}

#: Depth ascending so the default (depth 2) comes first and survives budget truncation.
TABPFN_REL_LOCAL_SPACE = SearchSpace(
    default_overrides={**_DEFAULT_KNOBS, "max_depth": _MIN_DEPTH},
    fixed_grid=[
        {**_DEFAULT_KNOBS, "max_depth": d} for d in range(_MIN_DEPTH, DFS_MAX_DEPTH + 1)
    ],
)

#: Hosted-API variant; the API handles anchor-table text server-side, so text is on.
#: API fits consume service quota, so callers should use `n_trials=0` unless they
#: explicitly want to spend quota on tuning.
_CLIENT_DEFAULT_KNOBS = {
    **_DEFAULT_KNOBS,
    "tfm": "tabpfn-v3-api",
    "with_text": True,
}
TABPFN_REL_CLIENT_SPACE = SearchSpace(
    default_overrides={**_CLIENT_DEFAULT_KNOBS, "max_depth": _MIN_DEPTH},
    fixed_grid=[
        {**_CLIENT_DEFAULT_KNOBS, "max_depth": d}
        for d in range(_MIN_DEPTH, DFS_MAX_DEPTH + 1)
    ],
)


class TabPFNRelModel(RelArenaModel):
    """Shared `tabpfn-rel` implementation; the registered variants pin the backend."""

    #: Depth at which the cached DFS matrix is computed (shallower configs slice it).
    MAX_DEPTH = DFS_MAX_DEPTH

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
        """Build DFS features, apply the enabled extras, then fit the TFM.

        The explicitly constructed `CacheConfig` determines whether DFS reads,
        fills, or computes privately; this model does not override that policy.
        """
        self._tfm = self.config.get("tfm", "tabpfn-v3")
        self._depth = int(self.config.get("max_depth", _MIN_DEPTH))
        self._check_text_support()
        self._features = FeaturePipeline(self.config)
        context = ContextStrategy.from_config(self.config)
        # The history pool and the recency contexts use the full train table; only the
        # TFM's context is capped, downstream by the context strategy.
        self._history_table = train_table if task.time_col else None

        df = self._dfs(task, db, train_table)
        if df.shape[1] == 0:
            raise ValueError(f"DFS produced no features at depth {self._depth}.")
        df = self._features.fit_transform(df, task, train_table, db)
        cutoff = train_table.df[task.time_col].to_numpy() if task.time_col else None
        self._fitted = context.fit(
            df,
            train_table.df[task.target_col],
            task.task_type,
            tfm=self._tfm,
            seed=seed,
            context_time=cutoff,
        )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Build + extend DFS features for `table` and return TFM predictions."""
        df = self._dfs(task, db, table)
        df = self._features.transform(df, task, table, db)
        return predict_tfm(self._fitted, df)

    def warm_cache(
        self, task: EntityTask, db: Database, train_table: Table, eval_table: Table
    ) -> None:
        """Populate the feature cache for this config's fit + predict, without the TFM.

        Runs exactly the `build_dfs_features` + feature-pipeline calls `fit` and
        `predict` make (same cache keys), so a later eval reads them instead of
        recomputing. Construct the model with an explicit fill config; needs only CPU.
        The shared public command is `relarena.featurization.warm_cache`.
        """
        self._tfm = self.config.get("tfm", "tabpfn-v3")
        self._depth = int(self.config.get("max_depth", _MIN_DEPTH))
        self._check_text_support()
        self._features = FeaturePipeline(self.config)
        self._history_table = train_table if task.time_col else None
        train_feats = self._dfs(task, db, train_table)
        self._features.fit_transform(train_feats, task, train_table, db)
        eval_feats = self._dfs(task, db, eval_table)
        self._features.transform(eval_feats, task, eval_table, db)

    def _check_text_support(self) -> None:
        """Raise if `with_text` is set on a TFM that does not support text."""
        if self.config.get("with_text") and not TFM_REGISTRY[self._tfm].supports_text:
            raise ValueError(
                f"with_text is set but TFM {self._tfm!r} does not support text "
                f"features; pick a text-capable TFM or unset with_text."
            )

    def _dfs(self, task: EntityTask, db: Database, table: Table) -> Any:
        features, _ = build_dfs_features(
            task,
            db,
            table,
            depth=self._depth,
            max_depth=self.MAX_DEPTH,
            history_table=self._history_table,
            keep_anchor_columns=True,
            cache=self.cache,
            run_identity=self.run_identity,
        )
        return features


@register_model(search_space=TABPFN_REL_LOCAL_SPACE)
class TabPFNRelLocalModel(TabPFNRelModel):
    """`tabpfn-rel` on the local TabPFN v3, without text features."""

    name = "tabpfn-rel-local"


@register_model(search_space=TABPFN_REL_CLIENT_SPACE)
class TabPFNRelClientModel(TabPFNRelModel):
    """`tabpfn-rel` through the hosted TabPFN API, with raw anchor-table text."""

    name = "tabpfn-rel-client"
