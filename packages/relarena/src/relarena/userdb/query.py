"""Python façade for the Relational Predictive Interface (RPI).

`PredictiveQuery` wraps the pieces validated separately — `UserEntityTask`
(SQL → labels), `RelBenchDatasetTask.from_objects` (splits), and
`predict_at` (label-less inference) — into a single object so the common
flow reads as `PredictiveQuery(spec).fit(model).predict()`.

`fit` runs RelArena's standard protocol by default: tune the model's search
space on the inner split (train→val, DB censored at `val_timestamp`), select
the best config by validation score, then perform the model's final-fit regime
on the outer split. For example, TabPFN-Rel sweeps its depth grid and picks the
best configuration rather than running one default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from relbench.base import EntityTask

from relarena.cache import CacheConfig, resolve_cache_config
from relarena.dataset import RelBenchDatasetTask, concat_tables
from relarena.identity import (
    RunIdentity,
    database_schema_fingerprint,
    task_spec_fingerprint,
)
from relarena.model import RelArenaModel
from relarena.registry import registry
from relarena.runner import select_best
from relarena.search_space import TaskStats, resolve_search_space
from relarena.system import RelArenaSystem
from relarena.tuner import tune as run_tuning
from relarena.userdb._schema import load_schema, validate
from relarena.userdb.ingest import DatabaseSpec, build_dataset
from relarena.userdb.predict import EntitySelector, predict_at
from relarena.userdb.spec import PredictiveTaskSpec
from relarena.userdb.task import UserEntityTask

#: JSON Schema for a task YAML; the single source of truth for its accepted shape.
_TASK_SCHEMA = load_schema("task.schema.json")


class PredictiveQuery:
    """A user task over a database: fit on historical labels, predict the future."""

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
        self._source = RelBenchDatasetTask.from_objects(
            dataset,
            UserEntityTask(dataset, task),
            dataset_name="user",
            run_identity=identity,
        )
        self.task: EntityTask = self._source.task
        #: The task's prediction target, read by `predict`.
        self._at_timestamp = task.at_timestamp
        self._entities = task.entities
        self._model: RelArenaModel | None = None
        self._cache = CacheConfig(directory=None, on_miss="compute")
        self._identity = identity
        self._warned_schema_only_cache = False
        #: Tuning trials and the selected config from the most recent `fit`.
        self.trials: list | None = None
        self.config: dict | None = None

    def fit(
        self,
        model: str,
        *,
        n_trials: int = 10,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> PredictiveQuery:
        """Fit the registered model named `model`; returns self so `predict` chains.

        With `n_trials > 0` (default 10): tune the model's search space on the
        inner split (train→val, DB censored at `val_timestamp`), select the best
        config by validation score, then perform its final fit on the outer split.
        For a sampled space, `n_trials` requests that many random samples in
        addition to the default; for a fixed grid, it caps the ordered grid.
        With `n_trials == 0`, skip tuning and fit the model's default config.

        `cache_dir` is a local directory that caches DFS features across tuning,
        the final fit, and later `predict`, useful for repeated runs on a large
        custom database. Omit it to fall back to `RELARENA_CACHE_DIR`, or to use
        no persistent cache when that variable is unset. See `relarena.cache`.
        """
        # Local import: importing the model package runs every built-in model's
        # registration, pulling in heavy/optional deps (torch via tabpfn, lightgbm)
        # at import time; kept out of `import relarena.userdb`, needed only here.
        import relarena.models  # noqa: F401

        cache = resolve_cache_config(cache_dir, on_miss="fill")
        self._warn_schema_only_cache(cache)
        self._cache = cache
        model_cls = registry.get(model)
        if isinstance(model_cls, type) and issubclass(model_cls, RelArenaSystem):
            raise TypeError(
                f"'{model}' is a RelArenaSystem. PredictiveQuery requires a "
                "RelArenaModel because it fits once and predicts at a later, "
                "caller-selected timestamp."
            )
        search_space = registry.search_space(model)

        # fill: on a custom DB the store starts empty, so build it as we go (the
        # tuning trials + refit then reuse it); a later run reads what this built.
        if n_trials > 0:
            self.trials = run_tuning(
                model_cls,
                search_space,
                self.task,
                self._source.inner_split(),
                n_trials=n_trials,
                seed=seed,
                cache=cache,
                run_identity=self._identity.for_phase("inner"),
            )
            self.config = select_best(self.trials, self._source.metric).config
        else:
            # Resolve a factory search space (e.g. relgt builds its grid from
            # TaskStats) before reading its defaults; a plain SearchSpace is
            # returned unchanged.
            self.trials = None
            stats = TaskStats(
                num_train_nodes=len(self._source.inner_split().train_table.df)
            )
            space = resolve_search_space(search_space, stats)
            self.config = dict(space.default_overrides)
        fitted = model_cls(
            self.config,
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
        self._model = fitted
        return self

    def precompute_cache(self, cache_dir: str | Path) -> str | Path:
        """Build the shared DFS feature cache on CPU; return `cache_dir`.

        Featurizes exactly what `fit` reads (the inner split for tuning, the outer
        refit split) in fill mode, so a later `fit(model, cache_dir=...)` reads the DFS
        instead of recomputing it. For a large custom database, run this on a big CPU
        node, then `fit` on the GPU pointing at the same `cache_dir`; `predict` fills
        its own anchor features on top. The resulting artifacts are shared by all DFS
        models.
        """
        from relarena.featurization.warm_cache import warm_dfs_cache

        cache = resolve_cache_config(cache_dir, on_miss="fill")
        self._warn_schema_only_cache(cache)
        warm_dfs_cache(self._source, cache)
        return cache_dir

    def predict(self, *, cache_dir: str | Path | None = None) -> pd.DataFrame:
        """Predict label-less rows for the task's configured target.

        The target is owned by the task spec: `at_timestamp` (the prediction anchor,
        defaulting to `test_timestamp`) and `entities` (defaulting to all). In line
        with the RelBench protocol, the feature database remains frozen at the
        task's `test_timestamp`; a later anchor changes the prediction seed but does
        not expose rows written after that cutoff.

        `entities` are the user's original ids; they are translated to the internal
        reindexed ids for scoring, and the returned `entity_col` is translated back
        to the original ids (for native RelBench datasets, which have no id map, ids
        pass through unchanged).

        `cache_dir` caches the prediction-time DFS features; it defaults to the one
        passed to `fit`.
        """
        fitted = self._model
        if fitted is None:
            raise RuntimeError("Call fit(...) first.")
        db = self._source._db
        test_timestamp = self._source._dataset.test_timestamp
        anchor = (
            test_timestamp
            if self._at_timestamp is None
            else pd.Timestamp(self._at_timestamp)
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
        entities = self._entities
        id_map = getattr(self._source._dataset, "pkey_maps", {}).get(
            self.task.entity_table
        )
        cache = (
            resolve_cache_config(cache_dir, on_miss="fill")
            if cache_dir is not None
            else self._cache
        )
        fitted.cache = cache
        fitted.run_identity = self._identity.for_phase("predict")
        self._warn_schema_only_cache(cache)
        preds = predict_at(
            fitted, self.task, db, anchor, self._to_internal_ids(entities, id_map)
        )
        if id_map is not None:
            to_original = pd.Series(id_map.index, index=id_map.to_numpy())
            preds[self.task.entity_col] = preds[self.task.entity_col].map(to_original)
        return preds

    def compute_test_labels(
        self, *, data_end_timestamp: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Compute labels for the historical test windows from the full database.

        Pass `data_end_timestamp` when the database is known to be complete only
        through a particular date (for example, for a partial or sparse extract).
        If omitted, the latest timestamp present anywhere in the database is used.
        Raises when that cutoff does not cover every test window's label horizon.
        """
        dataset = self._source._dataset
        full_db = dataset.get_db(upto_test_timestamp=False)
        available_until = (
            full_db.max_timestamp
            if data_end_timestamp is None
            else pd.Timestamp(data_end_timestamp)
        )
        required_until = (
            dataset.test_timestamp + self.task.timedelta * self.task.num_eval_timestamps
        )
        if available_until < required_until:
            raise ValueError(
                "Cannot compute complete test labels: the configured test windows "
                f"require data through {required_until}, but the database is "
                f"known only through {available_until}."
            )

        timestamps = pd.date_range(
            start=dataset.test_timestamp,
            periods=self.task.num_eval_timestamps,
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

    def _warn_schema_only_cache(self, cache: CacheConfig) -> None:
        if (
            cache.directory is not None
            and self._identity.data_version is None
            and not self._warned_schema_only_cache
        ):
            warnings.warn(
                "Persistent PredictiveQuery caching has no data_version; keys may "
                "fall back to a schema-only fingerprint. Pass data_version=... and "
                "bump it when row content changes.",
                stacklevel=3,
            )
            self._warned_schema_only_cache = True

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
    """A predictive task plus the database it runs against.

    Bundles the two halves a run needs - the `task` (label SQL, split timestamps,
    prediction target) and its `database` (table schema) - into one object. The
    solver choice (which model, how many tuning trials, the seed) is not part of the
    spec; it is passed to `PredictiveQuery.fit` at call time, so the same spec
    can run against several models. Load one from a task YAML via `from_yaml`.
    """

    database: DatabaseSpec
    task: PredictiveTaskSpec

    @classmethod
    def from_yaml(
        cls, path: str, *, data_dir: str | None = None
    ) -> PredictiveQuerySpec:
        """Load a spec from a task YAML file.

        The task YAML holds the task fields (label `query`, `entity_col`,
        `target_col`, `task_type`, `timedelta`), the `val_timestamp` /
        `test_timestamp` split cutoffs, an optional prediction target (`entities` /
        `at_timestamp`), and a `database` field naming the database YAML. The
        database path is resolved relative to the task file's directory (absolute
        paths used as-is); its tables' data files resolve against `data_dir`.
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
