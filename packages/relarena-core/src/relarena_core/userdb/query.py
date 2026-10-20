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

        Args:
            spec: The database and task; the task's split cutoffs define the splits.
            data_version: Label for this copy of the data, recorded in run identities.

        Raises:
            ValueError: If the validation window ends after the last row at or
                before `test_timestamp`.
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
        # Built first so RelBench's own check of the val/test gap reports first.
        entity_task = UserEntityTask(dataset, task)
        last_row = dataset.get_db().max_timestamp
        val_end = task.val_timestamp + task.timedelta
        if val_end > last_row:
            raise ValueError(
                f"The validation window ends at {val_end}, after the last row at or "
                f"before test_timestamp ({last_row}), so its labels would be "
                "incomplete. Set val_timestamp at least one timedelta before "
                f"{last_row}."
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
            entity_task,
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
        """Load a database and training task from YAML.

        Args:
            path: Task YAML; see `PredictiveQuerySpec.from_yaml`.
            data_dir: Base directory for the tables' data files.
            data_version: Label for this copy of the data, recorded in run identities.
        """
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

        Args:
            model: Registered model name, e.g. `"constant-global"` or
                `"tabpfn-rel-client-latest"`.
            n_trials: Tuning trials on the inner split (train→val, database
                censored at `val_timestamp`); the best config is then refit on the
                outer split. A sampled space draws this many configs besides the
                default; a fixed grid is capped at this length. `0` fits the
                default config.
            seed: Random seed for tuning and fitting.
            cache_dir: Local directory caching DFS features across tuning, the
                final fit and `predict`. `None` uses `RELARENA_CACHE_DIR`, or no
                persistent cache when it is unset.

        Returns:
            The fitted predictor, with tuning results in `trials`.
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
        refit split), so a later `fit(model, cache_dir=...)` reads the DFS instead of
        recomputing it. For a large custom database, run this on a big CPU node, then
        `fit` on the GPU. The artifacts are shared by all DFS models.

        Args:
            cache_dir: Local directory to fill; pass the same one to `fit`.
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

        Only complete test windows are returned. Rows for entities missing from the
        frozen database are dropped with a warning.

        Args:
            data_end_timestamp: Last date with complete data. `None` uses the
                latest timestamp in any table.

        Returns:
            The test rows: time, entity (original ids) and target columns.

        Raises:
            ValueError: If the first test window is not fully covered by the data.
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
        num_rows = len(labels.df)
        labels = (
            self.task.filter_dangling_entities(labels)
            .df[[self.task.time_col, self.task.entity_col, self.task.target_col]]
            .copy()
        )
        if dropped := num_rows - len(labels):
            warnings.warn(
                f"Dropped {dropped} test label row(s) whose entities are missing "
                f"from the database frozen at test_timestamp {dataset.test_timestamp} "
                "(created after it, or without a timestamp), so the model cannot "
                "score them.",
                UserWarning,
                stacklevel=2,
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
        """Yield each test timestamp and its labeled entity IDs.

        Args:
            labels: Test rows from `compute_test_labels`.

        Yields:
            `(timestamp, entity_ids)` pairs, ready for one `PredictiveQuery` each.
        """
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
    """Entities and anchor time to score with a fitted predictor.

    Attributes:
        entities: `"all"` or a sequence of original entity ids.
        at_timestamp: Anchor time, or `"test_timestamp"` for the context's test cutoff.
    """

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
        """Predict original entity IDs with features frozen at the context cutoff.

        Args:
            query: The entities and anchor timestamp to predict.
            cache_dir: Feature cache; `None` reuses the one passed to `fit`.

        Returns:
            One row per entity: the time and entity columns and `<target_col>_pred`.
        """
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

        Args:
            path: Task YAML with the label `query`, task fields, split cutoffs and
                a `database` path, relative to this file unless absolute.
            data_dir: Base directory for the tables' data files. `None` uses the
                database YAML's directory.
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
