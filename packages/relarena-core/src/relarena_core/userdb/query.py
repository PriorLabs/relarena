"""Training contexts, fitted predictors, and explicit prediction requests."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from relbench.base import EntityTask

from relarena_core.cache import CacheConfig, resolve_cache_config
from relarena_core.dataset import TaskSource, concat_tables
from relarena_core.discovery import discover_models
from relarena_core.identity import (
    RunIdentity,
    database_schema_fingerprint,
    task_spec_fingerprint,
)
from relarena_core.model import RelArenaModel
from relarena_core.registry import registry
from relarena_core.results import TrialResult
from relarena_core.search_space import TaskStats, resolve_search_space
from relarena_core.selection import select_best
from relarena_core.system import RelArenaSystem
from relarena_core.tuner import tune as run_tuning
from relarena_core.userdb._schema import load_schema, validate
from relarena_core.userdb.ingest import DatabaseSpec, build_dataset
from relarena_core.userdb.predict import EntitySelector, predict_at
from relarena_core.userdb.spec import PredictiveTaskSpec
from relarena_core.userdb.task import UserEntityTask

#: JSON Schema for a task YAML; the single source of truth for its accepted shape.
_TASK_SCHEMA = load_schema("task.schema.json")


class PredictiveContext:
    """A database and training task shared by fitted predictors."""

    def __init__(
        self, spec: PredictiveQuerySpec, *, data_version: str | None = None
    ) -> None:
        """Build the split source from a spec's database and SQL task.

        The task owns the split cutoffs, so the dataset is built here from the
        task's `val_timestamp` / `test_timestamp` rather than passed in separately.
        """
        database, task = spec.database, spec.task
        # Dissonance we live with: RelBench treats the split cutoffs as part of the
        # Dataset (its split reads dataset.val_timestamp / test_timestamp), while we
        # model them as part of the task. Not ideal, but building the dataset here
        # from the task keeps the task the single source of truth - the alternative,
        # taking a prebuilt dataset, duplicates the cutoffs and lets them drift.
        dataset = build_dataset(
            database,
            val_timestamp=task.val_timestamp,
            test_timestamp=task.test_timestamp,
        )
        identity = RunIdentity(
            dataset="user",
            dataset_fingerprint=database_schema_fingerprint(dataset._db),
            task=f"{task.entity_table}-{task.target_col}",
            task_fingerprint=task_spec_fingerprint(task),
            data_version=data_version,
        )
        self._source = TaskSource.from_objects(
            dataset,
            UserEntityTask(dataset, task),
            dataset_name="user",
            run_identity=identity,
        )
        self.task: EntityTask = self._source.task
        self._identity = identity
        self._warned_schema_only_cache = False

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        data_dir: str | Path | None = None,
        data_version: str | None = None,
    ) -> PredictiveContext:
        """Load a database and training task from YAML."""
        spec = PredictiveQuerySpec.from_yaml(path, data_dir=data_dir)
        return cls(spec, data_version=data_version)

    def fit(
        self,
        model: str,
        *,
        n_trials: int = 10,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> FittedPredictor:
        """Fit a registered model and return its independent fitted state.

        With `n_trials > 0` (default 10): tune the model's search space on the
        inner split (train→val, DB censored at `val_timestamp`), select the best
        config by validation score, then perform its final fit on the outer split.
        For a sampled space, `n_trials` requests that many random samples in
        addition to the default; for a fixed grid, it caps the ordered grid.
        With `n_trials == 0`, skip tuning and fit the model's default config.
        A plan containing only the default config also skips tuning.

        `cache_dir` is a local directory that caches DFS features across tuning,
        the final fit, and later `predict`, useful for repeated runs on a large
        custom database. Omit it to fall back to `RELARENA_CACHE_DIR`, or to use
        no persistent cache when that variable is unset. See `relarena_core.cache`.
        """
        discover_models(refresh=False)

        cache = resolve_cache_config(cache_dir, on_miss="fill")
        self._warn_schema_only_cache(cache)
        model_cls = registry.get(model)
        if isinstance(model_cls, type) and issubclass(model_cls, RelArenaSystem):
            raise TypeError(
                f"'{model}' is a RelArenaSystem. PredictiveContext requires a "
                "RelArenaModel because it fits once and predicts at a later, "
                "caller-selected timestamp."
            )
        search_space = registry.search_space(model)
        if callable(search_space):
            stats = TaskStats(
                num_train_nodes=len(self._source.inner_split().train_table.df)
            )
            search_space = resolve_search_space(search_space, stats)
        grid = search_space.fixed_grid
        default_only = (
            n_trials == 0
            or not search_space.is_tunable
            or (
                grid is not None and grid[:n_trials] == [search_space.default_overrides]
            )
        )

        # fill: on a custom DB the store starts empty, so build it as we go (the
        # tuning trials + refit then reuse it); a later run reads what this built.
        if n_trials > 0 and not default_only:
            trials = run_tuning(
                model_cls,
                search_space,
                self.task,
                self._source.inner_split(),
                n_trials=n_trials,
                seed=seed,
                cache=cache,
                run_identity=self._identity.for_phase("inner"),
            )
            config = select_best(trials, self._source.metric).config
        else:
            trials = None
            config = dict(search_space.default_overrides)
        fitted = model_cls(
            config,
            cache=cache,
            run_identity=self._identity.for_phase("outer"),
        )

        # Refit the chosen config on the outer split, matching the model's
        # final-fit regime (train+val union for refit_on_full_data models).
        outer = self._source.outer_split()
        if model_cls.refit_on_full_data:
            train_table, val_table = (
                concat_tables(outer.train_table, outer.val_table),
                None,
            )
        else:
            train_table, val_table = outer.train_table, outer.val_table

        fitted.fit(self.task, outer.db_state, train_table, val_table, seed=seed)
        return FittedPredictor(self, fitted, cache, trials, config)

    def precompute_cache(self, cache_dir: str | Path) -> str | Path:
        """Build the shared DFS feature cache on CPU; return `cache_dir`.

        Featurizes exactly what `fit` reads (the inner split for tuning, the outer
        refit split) in fill mode, so a later `fit(model, cache_dir=...)` reads the DFS
        instead of recomputing it. For a large custom database, run this on a big CPU
        node, then `fit` on the GPU pointing at the same `cache_dir`; `predict` fills
        its own anchor features on top. The resulting artifacts are shared by all DFS
        models.
        """
        from relarena_core.featurization.warm_cache import warm_dfs_cache

        cache = resolve_cache_config(cache_dir, on_miss="fill")
        self._warn_schema_only_cache(cache)
        warm_dfs_cache(self._source, cache)
        return cache_dir

    def compute_test_labels(
        self, *, data_end_timestamp: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Compute labels for the historical test windows from the full database.

        Pass `data_end_timestamp` when the database is known to be complete only
        through a particular date (for example, for a partial or sparse extract).
        If omitted, the latest timestamp present anywhere in the database is used.
        Only complete test windows are returned; at least one must be available.
        """
        dataset = self._source._dataset
        full_db = dataset.get_db(upto_test_timestamp=False)
        available_until = (
            full_db.max_timestamp
            if data_end_timestamp is None
            else pd.Timestamp(data_end_timestamp)
        )
        required_until = dataset.test_timestamp + self.task.timedelta
        if available_until < required_until:
            raise ValueError(
                "Cannot compute complete test labels: the configured test windows "
                f"require data through {required_until}, but the database is "
                f"known only through {available_until}."
            )

        timestamps = pd.date_range(
            start=dataset.test_timestamp,
            end=min(
                dataset.test_timestamp
                + self.task.timedelta * (self.task.num_eval_timestamps - 1),
                available_until - self.task.timedelta,
            ),
            freq=self.task.timedelta,
        )
        labels = self.task.make_table(full_db, timestamps)
        labels = (
            self.task.filter_dangling_entities(labels)
            .df[[self.task.time_col, self.task.entity_col, self.task.target_col]]
            .copy()
        )
        id_map = getattr(self._source._dataset, "pkey_maps", {}).get(
            self.task.entity_table
        )
        if id_map is not None:
            to_original = pd.Series(id_map.index, index=id_map.to_numpy())
            labels[self.task.entity_col] = labels[self.task.entity_col].map(to_original)
        return labels

    def group_test_entities(
        self, labels: pd.DataFrame
    ) -> Iterator[tuple[pd.Timestamp, list[Any]]]:
        """Yield each test timestamp and its labeled entity IDs."""
        for timestamp, rows in labels.groupby(self.task.time_col, sort=True):
            yield pd.Timestamp(timestamp), rows[self.task.entity_col].tolist()

    def _warn_schema_only_cache(self, cache: CacheConfig) -> None:
        if (
            cache.directory is not None
            and self._identity.data_version is None
            and not self._warned_schema_only_cache
        ):
            warnings.warn(
                "Persistent PredictiveContext caching has no data_version; keys may "
                "fall back to a schema-only fingerprint. Pass data_version=... and "
                "bump it when row content changes.",
                stacklevel=3,
            )
            self._warned_schema_only_cache = True


@dataclass(frozen=True, kw_only=True)
class PredictiveQuery:
    """Entities and anchor time to score with a fitted predictor."""

    entities: EntitySelector
    at_timestamp: str | pd.Timestamp

    def __post_init__(self) -> None:
        """Validate selections and normalize explicit timestamps."""
        if isinstance(self.entities, str):
            if self.entities != "all":
                raise ValueError("entities must be 'all' or a sequence of IDs.")
        else:
            try:
                object.__setattr__(self, "entities", tuple(self.entities))
            except TypeError:
                raise ValueError(
                    "entities must be 'all' or a sequence of IDs."
                ) from None
        if self.at_timestamp != "test_timestamp":
            timestamp = pd.Timestamp(self.at_timestamp)
            if pd.isna(timestamp):
                raise ValueError("at_timestamp must be a date or 'test_timestamp'.")
            object.__setattr__(self, "at_timestamp", timestamp)


@dataclass
class FittedPredictor:
    """A fitted model and its tuning results for one training context."""

    context: PredictiveContext
    _model: RelArenaModel
    _cache: CacheConfig
    trials: list[TrialResult] | None
    config: dict[str, Any]

    def predict(
        self, query: PredictiveQuery, *, cache_dir: str | Path | None = None
    ) -> pd.DataFrame:
        """Predict original entity IDs with features frozen at the context cutoff."""
        fitted = self._model
        db = self.context._source._db
        test_timestamp = self.context._source._dataset.test_timestamp
        anchor = (
            test_timestamp
            if query.at_timestamp == "test_timestamp"
            else pd.Timestamp(query.at_timestamp)
        )
        if anchor > test_timestamp:
            warnings.warn(
                f"Prediction anchor {anchor} is after test_timestamp "
                f"{test_timestamp}. RelArena follows the RelBench protocol, so "
                "the feature database remains frozen at test_timestamp and rows "
                "after that cutoff are not visible to the model.",
                UserWarning,
                stacklevel=2,
            )
        entities = query.entities
        id_map = getattr(self.context._source._dataset, "pkey_maps", {}).get(
            self.context.task.entity_table
        )
        cache = (
            resolve_cache_config(cache_dir, on_miss="fill")
            if cache_dir is not None
            else self._cache
        )
        fitted.cache = cache
        fitted.run_identity = self.context._identity.for_phase("predict")
        self.context._warn_schema_only_cache(cache)
        preds = predict_at(
            fitted,
            self.context.task,
            db,
            anchor,
            self._to_internal_ids(entities, id_map),
        )
        if id_map is not None:
            to_original = pd.Series(id_map.index, index=id_map.to_numpy())
            preds[self.context.task.entity_col] = preds[
                self.context.task.entity_col
            ].map(to_original)
        return preds

    @staticmethod
    def _to_internal_ids(
        entities: EntitySelector, id_map: pd.Series | None
    ) -> EntitySelector:
        """Map explicit original entity ids to internal reindexed ids.

        `"all"` and native-dataset (no map) inputs pass through. Ids absent from
        the database are dropped with a warning (they can't be scored).
        """
        if isinstance(entities, str) or id_map is None:
            return entities
        if not hasattr(entities, "__iter__"):
            raise ValueError(
                f"entities must be 'all' or a list of ids; got scalar {entities!r} "
                "- wrap a single id in a list, e.g. [5]."
            )
        requested = list(entities)
        mapped = id_map.reindex(requested).to_numpy()
        unknown = [e for e, idx in zip(requested, mapped) if pd.isna(idx)]
        if unknown:
            warnings.warn(
                f"Dropping {len(unknown)} requested entity id(s) absent from the "
                f"database: {unknown}",
                stacklevel=3,
            )
        return [int(idx) for idx in mapped if not pd.isna(idx)]


@dataclass(frozen=True)
class PredictiveQuerySpec:
    """A training task and database specification."""

    database: DatabaseSpec
    task: PredictiveTaskSpec

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, data_dir: str | Path | None = None
    ) -> PredictiveQuerySpec:
        """Load a spec from a task YAML file.

        The task YAML holds the task fields (label `query`, `entity_col`,
        `target_col`, `task_type`, `timedelta`), the `val_timestamp` /
        `test_timestamp` split cutoffs and a `database` field naming the database
        YAML. The database path is resolved relative to the task file's directory
        (absolute paths used as-is); its tables' data files resolve against `data_dir`.
        """
        raw = yaml.safe_load(Path(path).read_text())
        validate(raw, _TASK_SCHEMA, kind="task")
        db_ref = raw["database"]
        db_path = (
            db_ref if Path(db_ref).is_absolute() else str(Path(path).parent / db_ref)
        )
        database = DatabaseSpec.from_yaml(db_path, data_dir=data_dir)
        task = PredictiveTaskSpec(**{k: v for k, v in raw.items() if k != "database"})
        return cls(database=database, task=task)
