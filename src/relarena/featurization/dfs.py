"""Deep Feature Synthesis featurization (the RDBLearn recipe).

Flattens the relational database into a single feature table by aggregating over
the foreign-key graph up to a bounded *depth* (counts, means, std, ... of related
rows), via the `fastdfs` library (its `dfs2sql` engine). Unlike the entity-only
recipe, this materializes multi-hop neighbor features.

The RDB is built from the harness's *test-time-censored* `db`
(`get_db(upto_test_timestamp=True)`) — never a freshly loaded dataset, which
would leak post-cutoff rows. DFS gets the per-row `cutoff_time_column` so each
entity only sees history up to its prediction time.

**RDB transform pipeline.** Before DFS, the RDB goes through a transform
pipeline that turns datetime columns into epochtime floats — without this,
fastdfs types timestamps as non-numeric and the aggregation primitives emit *no*
features from them (no recency signals). Text columns are dropped. TODO: numeric
values mis-stored as strings (common in dirty real-world tables) are dropped along
with genuine text; coerce string columns that parse as numbers before dropping.

**Temporal diff.** After DFS, absolute `*_epochtime` feature columns become
differences against each row's cutoff time: absolute timestamps don't generalize
across a temporal train/test split; time-since-cutoff does.

**Target history.** `build_dfs_features` optionally injects a label-history
table (`_RDBL_target_history`) into the RDB before DFS, so DFS derives
aggregates of *past targets* per entity. The strictly-less-than cutoff join
keeps this leak-free: an anchor row never sees a label at or after its own
timestamp.

**Depth cache.** DFS feature sets are *nested*: the columns produced at depth `d`
are a subset of those at depth `d+1`. So instead of recomputing DFS for every
depth in a sweep, we compute the matrix **once at `max_depth`** per (split,
history) pair and, for a request at depth `d`, slice to columns whose
feature-depth is `<= d` — which reproduces a depth-`d` DFS run exactly. The
cache (`_CACHE`) is module-level so it is shared across the fresh model
instances the harness builds per config; it is scoped to one `db` by identity
(reset when a new `db` appears).

Source: adapts the RDBLearn approach (https://github.com/HKUSHXLab/rdblearn);
the depth cache here is new (RDBLearn recomputes DFS per fit).
"""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable

import pandas as pd
from relbench.base import Database, EntityTask, Table

from relarena.cache import CacheConfig, cache_key
from relarena.checksums import database_checksum, table_checksum
from relarena.featurization._columns import type_columns
from relarena.featurization.cache import cached_frame
from relarena.identity import RunIdentity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastdfs import RDB

#: Deepest shared DFS matrix built by the warmer and sliced by model depth grids.
#: This is part of artifact identity, so models and warmers must not define it
#: independently.
DFS_MAX_DEPTH = 4

#: Name of the injected label-history table.
TARGET_HISTORY_TABLE_NAME = "_RDBL_target_history"

#: Bumped when DFS-building or the RDB-transform pipeline changes in a way that
#: should invalidate persisted feature matrices — the disk cache key pins the data
#: content and build params, not the code that turns one into the other.
_DFS_CACHE_VERSION = 2


def _source_identity_segments(
    db: Database, task: EntityTask, identity: RunIdentity | None
) -> tuple[str, ...]:
    """Return source-data segments, with direct-input fallbacks."""
    dataset = "direct" if identity is None else identity.dataset
    db_fingerprint = (
        None if identity is None else identity.dataset_fingerprint
    ) or f"{int(database_checksum(db)):016x}"
    task_name = (
        f"{task.entity_table}-{task.target_col}"
        if identity is None or identity.task is None
        else identity.task
    )
    task_fingerprint = (
        None if identity is None else identity.task_fingerprint
    ) or f"{int(table_checksum(_task_shape(task))):016x}"
    segments = [
        f"{dataset}@{db_fingerprint}",
        f"{task_name}@{task_fingerprint}",
    ]
    if identity is not None and identity.data_version is not None:
        segments.append(f"data-{_stable_text_digest(identity.data_version)}")
    return tuple(segments)


def _phase_segment(identity: RunIdentity | None) -> str:
    """Return the protocol phase used by phase-dependent matrix artifacts."""
    phase = "direct" if identity is None or identity.phase is None else identity.phase
    return f"phase-{phase}"


def _stable_text_digest(value: str) -> str:
    import hashlib

    return hashlib.blake2s(value.encode(), digest_size=8).hexdigest()


def _task_shape(task: EntityTask) -> Table:
    """Represent task fields used by DFS as a tiny checksum-compatible table."""
    return Table(
        df=pd.DataFrame(
            {
                "entity_table": [task.entity_table],
                "entity_col": [task.entity_col],
                "time_col": [task.time_col],
                "target_col": [task.target_col],
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col=None,
        time_col=None,
    )


def _anchor_checksum(table: Table, target_col: str) -> int:
    """Checksum the anchor rows DFS consumes, excluding passthrough targets."""
    columns = [column for column in table.df if column != target_col]
    return int(
        table_checksum(
            Table(
                df=table.df[columns],
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
        )
    )


def _dfs_cache_key(
    db: Database,
    task: EntityTask,
    table: Table,
    history_table: Table | None,
    max_depth: int,
    identity: RunIdentity | None,
) -> PurePosixPath:
    """Complete matrix key from DFS's actual data and semantic inputs."""
    history = (
        "none"
        if history_table is None
        else f"full-{int(table_checksum(history_table)):016x}"
    )
    return cache_key(
        "dfs",
        f"v{_DFS_CACHE_VERSION}",
        *_source_identity_segments(db, task, identity),
        _phase_segment(identity),
        f"anchor-{_anchor_checksum(table, task.target_col):016x}",
        f"history-{history}",
        f"max-depth-{max_depth}",
        "matrix.parquet",
    )


def _dfs_depth_map_key(
    db: Database,
    task: EntityTask,
    history_table: Table | None,
    max_depth: int,
    identity: RunIdentity | None,
) -> PurePosixPath:
    """Complete schema-depth key, intentionally independent of anchor rows."""
    history_mode = "full" if history_table is not None else "none"
    return cache_key(
        "dfs",
        f"v{_DFS_CACHE_VERSION}",
        *_source_identity_segments(db, task, identity),
        f"history-{history_mode}",
        f"max-depth-{max_depth}",
        "depths.parquet",
    )


def _cached_depth_map(
    cache: CacheConfig,
    key: str | PurePosixPath | Callable[[], str | PurePosixPath],
    compute: Callable[[], dict[str, int]],
) -> dict[str, int]:
    """Persist/load the depth map through the DFS-owned parquet codec."""

    def _as_frame() -> pd.DataFrame:
        d = compute()
        return pd.DataFrame({"feature": list(d.keys()), "depth": list(d.values())})

    def _key() -> str:
        return str(key() if callable(key) else key)

    df = cached_frame(cache, _key, _as_frame)
    return dict(zip(df["feature"], df["depth"].astype(int)))


def _build_rdb(db: Database) -> "RDB":
    """Construct a fastdfs `RDB` from a relbench `Database` (untransformed).

    Keys are coerced to strings (fastdfs' `safe_convert_to_string`): relbench
    primary/foreign keys are commonly integer-typed, and the dfs2sql engine requires
    consistent string key dtypes across each parent PK and the child FKs to it.

    The transform pipeline (`_transform_rdb`) is applied separately, *after*
    any target-history table is added, so the history columns get the same
    datetime/text handling.
    """
    from fastdfs import create_rdb
    from fastdfs.utils.type_utils import safe_convert_to_string

    table_dict = db.table_dict

    tables: dict[str, pd.DataFrame] = {}
    primary_keys: dict[str, str] = {}
    time_columns: dict[str, str] = {}
    foreign_keys: list[tuple[str, str, str, str]] = []

    key_cols: dict[str, set[str]] = {name: set() for name in table_dict}
    for name, t in table_dict.items():
        if t.pkey_col:
            key_cols[name].add(t.pkey_col)
        for child_col in t.fkey_col_to_pkey_table:
            key_cols[name].add(child_col)

    for name, t in table_dict.items():
        df = t.df.copy()
        for col in key_cols[name]:
            if col in df.columns:
                df[col] = safe_convert_to_string(df[col])
        tables[name] = df
        if t.pkey_col:
            primary_keys[name] = t.pkey_col
        if t.time_col:
            time_columns[name] = t.time_col
        for child_col, parent_table in t.fkey_col_to_pkey_table.items():
            parent_pk = table_dict[parent_table].pkey_col
            foreign_keys.append((name, child_col, parent_table, parent_pk))

    return create_rdb(
        tables,
        name="relarena",
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        time_columns=time_columns,
    )


def _transform_rdb(rdb: "RDB") -> "RDB":
    """Apply the pre-DFS RDB transform pipeline.

    `FeaturizeDatetime` is the load-bearing step: the `*_epochtime` float
    columns it adds are what let the numeric aggregation primitives produce
    temporal features.
    """
    from fastdfs.transform import (
        CanonicalizeTypes,
        FeaturizeDatetime,
        FillMissingPrimaryKey,
        FilterColumn,
        HandleDummyTable,
        RDBTransformPipeline,
        RDBTransformWrapper,
    )

    pipeline = RDBTransformPipeline(
        [
            HandleDummyTable(),
            FillMissingPrimaryKey(),
            RDBTransformWrapper(FeaturizeDatetime(features=["epochtime"])),
            RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
            RDBTransformWrapper(CanonicalizeTypes()),
        ]
    )
    return pipeline(rdb)


def _history_frame(task: EntityTask, history_table: Table) -> pd.DataFrame:
    """The `(entity, time, target)` frame injected as the label-history table."""
    from fastdfs.utils.type_utils import safe_convert_to_string

    cols = [task.entity_col, task.time_col, task.target_col]
    df = history_table.df[cols].copy()
    df[task.entity_col] = safe_convert_to_string(df[task.entity_col])
    return df


def _add_history_table(rdb: "RDB", task: EntityTask, history_df: pd.DataFrame) -> "RDB":
    """Add `history_df` as `_RDBL_target_history` with an FK to the entity table.

    Registered with `time_column` so the DFS cutoff join only aggregates history
    rows strictly before each anchor's timestamp (no label leakage).
    """
    entity_pk = rdb.get_table_metadata(task.entity_table).primary_key
    rdb = rdb.add_table(
        dataframe=history_df,
        name=TARGET_HISTORY_TABLE_NAME,
        time_column=task.time_col,
        foreign_keys=[(task.entity_col, task.entity_table, entity_pk)],
    )
    rdb = rdb.canonicalize_key_types()
    rdb.validate_key_consistency()
    return rdb


def _shallow_frame_copy(rdb: "RDB") -> "RDB":
    """A copy of `rdb` whose table frames are `copy(deep=False)` copies."""
    return rdb.update_tables(
        tables={
            n: rdb.get_table_dataframe(n).copy(deep=False) for n in rdb.table_names
        },
        metadata={n: rdb.get_table_metadata(n) for n in rdb.table_names},
    )


def _temporal_diff(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    """Convert `*_epochtime` feature columns to time-until-cutoff differences.

    Drops epochtime columns whose name contains `std` (std of timestamps is
    meaningless) and replaces every other epochtime column with
    `{sanitized_name}_diff = cutoff_ns - value` (NaN propagates). No-op when
    the frame has no cutoff column to diff against.
    """
    epochtime_cols = [c for c in df.columns if "_epochtime" in c]
    if not epochtime_cols:
        return df
    df = df.copy()
    drop = [c for c in epochtime_cols if "std" in c.lower()]
    df = df.drop(columns=drop)
    keep = [c for c in epochtime_cols if c not in drop]
    if not keep or time_col is None or time_col not in df.columns:
        return df

    cutoff_ns = df[time_col].astype("datetime64[ns]").astype("int64").astype("float64")
    for col in keep:
        sanitized = col.replace("(", "_").replace(")", "").replace(".", "_")
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        df[f"{sanitized.strip('_')}_diff"] = cutoff_ns - df[col].astype("float64")
        df = df.drop(columns=[col])
    return df


def _calendar_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Append year/month/day/dayofweek decompositions of the cutoff timestamp."""
    ts = pd.to_datetime(df[time_col])
    df = df.copy()
    for name, values in [
        ("year", ts.dt.year),
        ("month", ts.dt.month),
        ("day", ts.dt.day),
        ("dayofweek", ts.dt.dayofweek),
    ]:
        df[f"{time_col}.{name}"] = values.astype("float64")
    return df


class _DepthCache:
    """Process-level DFS cache shared across model instances.

    The harness builds a fresh model per config (= per depth), so for the deepest
    DFS to be computed once and reused by shallower depths, the cache must live at
    module level. Holds, for one `db`:

      * the untransformed `RDB` (one `_build_rdb` per db);
      * transformed RDB *variants* keyed by history frame: the plain pipeline'd
        RDB (key `None`) and one per injected `_RDBL_target_history` frame;
      * the `{feature_name: depth}` map at `max_depth` per variant
        (schema-based);
      * the full `max_depth` feature matrix per (split dataframe, variant).

    Scoped to a single db by *identity* (`is`, holding the db reference so an
    object-id reuse after GC can't cause a false hit); every entry point calls
    `_scope`, which drops all memos when a new db appears. History variants hold a
    reference to their history frame for the same reason; split frames are kept
    alive by the harness.

    Implementation notes:
      * `compute` / `build` callables are *thunks*: the cache invokes them
        only on a miss, to produce the value to store. This keeps the cache
        agnostic of *how* each value is built.
      * Because scoping is by object identity (above), a *copy* of the db or of a
        split frame counts as new and is recomputed — wasted work, never a wrong
        result (a copy has identical data). Fine for now, since the harness holds
        stable objects for a whole run; revisit only if we ever key by content or
        run datasets concurrently in one process.
    """

    def __init__(self) -> None:
        self._db: Database | None = None
        self._raw_rdb: "RDB | None" = None
        #: history-frame key -> (history frame ref, transformed RDB variant)
        self._rdbs: dict[int | None, tuple[pd.DataFrame | None, "RDB"]] = {}
        self._depth_maps: dict[tuple[int | None, int, CacheConfig], dict[str, int]] = {}
        self._matrices: dict[
            tuple[Database, int, int | None, int, CacheConfig], pd.DataFrame
        ] = {}
        self.matrix_computations = 0  # instrumentation: # of max_depth DFS runs

    @staticmethod
    def _history_key(history: pd.DataFrame | None) -> int | None:
        return None if history is None else id(history)

    def _scope(self, db: Database) -> None:
        """Drop every memo when a new db comes into scope.

        Every public entry point scopes first. Eviction must not hang off
        `raw_rdb_for` alone: with a warm on-disk cache neither the matrix nor the
        depth map ever builds an RDB (`build_dfs_features` constructs it lazily),
        so that path is never reached — leaving the previous db's memos to
        accumulate *and* to be served to the next db.

        Scoping is by identity (`is`), holding the db reference so an address
        reused after GC can't cause a false hit.
        """
        if self._db is db:
            return
        self._db = db
        self._raw_rdb = None
        self._rdbs = {}
        self._depth_maps = {}
        self._matrices = {}

    def raw_rdb_for(self, db: Database) -> "RDB":
        """The untransformed RDB for `db` (resets the cache on a new db)."""
        self._scope(db)
        if self._raw_rdb is None:
            self._raw_rdb = _build_rdb(db)
        return self._raw_rdb

    def rdb_variant(
        self,
        db: Database,
        history: pd.DataFrame | None,
        build: Callable[["RDB"], "RDB"],
    ) -> "RDB":
        """The transformed RDB for `(db, history)`; `build` runs on a miss.

        `build` receives the untransformed base RDB and returns the transformed
        variant (with the history table added first, when `history` is given).
        The history frame reference is held so its `id` stays valid.
        """
        raw = self.raw_rdb_for(db)
        key = self._history_key(history)
        if key not in self._rdbs:
            self._rdbs[key] = (history, build(raw))
        return self._rdbs[key][1]

    def full_matrix(
        self,
        db: Database,
        source_df: pd.DataFrame,
        history: pd.DataFrame | None,
        max_depth: int,
        cache: CacheConfig,
        compute: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """The `max_depth` matrix for `(db, source_df, history)`; builds on a miss."""
        # `_scope` bounds this to the current db, so the key covers only what varies
        # within one: the split frame and history identities preserve fast reuse
        # across config instances, while max_depth and the explicit config prevent a
        # scratch computation from hiding a later fill, or a shallow matrix from
        # satisfying a deeper request.
        self._scope(db)
        key = (id(source_df), self._history_key(history), max_depth, cache)
        if key not in self._matrices:
            self.matrix_computations += 1
            self._matrices[key] = compute()
        return self._matrices[key]

    def depth_map(
        self,
        db: Database,
        history: pd.DataFrame | None,
        max_depth: int,
        cache: CacheConfig,
        compute: Callable[[], dict[str, int]],
    ) -> dict[str, int]:
        """The `{feature: depth}` map for `(db, history)`; `compute` on a miss.

        `db` scopes the memo: the map is keyed by feature *name*, so another
        database's map matches nothing and the depth filter in
        `build_dfs_features` would then keep every column.
        """
        self._scope(db)
        key = (self._history_key(history), max_depth, cache)
        if key not in self._depth_maps:
            self._depth_maps[key] = compute()
        return self._depth_maps[key]


#: Module-level cache, shared across all callers in the process.
_CACHE = _DepthCache()


def build_dfs_features(
    task: EntityTask,
    db: Database,
    table: Table,
    *,
    depth: int,
    max_depth: int,
    history_table: Table | None = None,
    keep_anchor_columns: bool = False,
    cache: CacheConfig | None = None,
    run_identity: RunIdentity | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build DFS features for a task's label `table` at the given `depth`.

    Computes the feature matrix once at `max_depth` (cached per split dataframe
    and history variant) and slices it to the columns at depth `<= depth`. Then
    converts absolute epochtime features to time-until-cutoff diffs (see
    `_temporal_diff`), drops identifier / timestamp / target columns and
    splits the rest into numeric + categorical.

    `history_table` (the *full, un-subsampled* train label table; train+val on
    refit) is injected as `_RDBL_target_history` so DFS derives past-label
    aggregates per entity; requires `task.time_col` (skipped otherwise).
    `keep_anchor_columns` keeps the entity key as a categorical feature and the
    cutoff timestamp as numeric + calendar features.

    Returns `(features_df, categorical_columns)` — same contract as
    `build_entity_features`; the caller freezes a consistent categorical encoding
    across splits.
    """
    from fastdfs import DFSConfig, compute_dfs_features
    from fastdfs.dfs import get_dfs_engine
    from fastdfs.utils.type_utils import safe_convert_to_string

    cache = cache or CacheConfig(directory=None, on_miss="compute")

    entity_table = db.table_dict[task.entity_table]
    key_mappings = {task.entity_col: f"{task.entity_table}.{entity_table.pkey_col}"}
    source_df = table.df

    if history_table is not None and task.time_col is None:
        # Without a cutoff column the history join can't be made leak-free.
        history_table = None
    history_df = None if history_table is None else _history_frame(task, history_table)
    # Cache key: the caller-owned table frame (stable identity); the derived
    # history_df is rebuilt per call, so its id would never hit.
    history_key_df = None if history_table is None else history_table.df

    def _build_variant(raw: "RDB") -> "RDB":
        rdb = raw
        if history_df is not None:
            rdb = _add_history_table(rdb, task, history_df)
        return _transform_rdb(rdb)

    # Built lazily: a warm cache (matrix + depth map both hit) needs no RDB at all.
    _rdb_holder: dict[str, "RDB"] = {}

    def _rdb() -> "RDB":
        if "rdb" not in _rdb_holder:
            _rdb_holder["rdb"] = _CACHE.rdb_variant(db, history_key_df, _build_variant)
        return _rdb_holder["rdb"]

    def _dfs_input() -> pd.DataFrame:
        # DFS feature columns depend only on the schema, so dropping the target (a
        # passthrough column) keeps the cached matrix consistent across splits.
        df = source_df.copy()
        if task.target_col in df.columns:
            df = df.drop(columns=[task.target_col])
        if task.entity_col in df.columns:
            df[task.entity_col] = safe_convert_to_string(df[task.entity_col])
        return df

    # fastdfs' dfs2sql engine materializes a DuckDB file at `cfg.engine_path`; left
    # unset it defaults to a `fastdfs_<uuid>.db` under the system tempdir that it
    # never deletes, so per-fit DFS calls (rdblearn / tabpfn-rel) leak
    # multi-GB files indefinitely. Pin it inside a TemporaryDirectory removed once the
    # matrix is built; only the computed frames outlive this block.
    with tempfile.TemporaryDirectory(prefix="relarena_dfs_") as engine_dir:
        cfg = DFSConfig(
            max_depth=max_depth, engine_path=str(Path(engine_dir) / "dfs.db")
        )

        def _compute_matrix() -> pd.DataFrame:
            return compute_dfs_features(
                _rdb(),
                _dfs_input(),
                key_mappings=key_mappings,
                cutoff_time_column=task.time_col,
                config=cfg,
            )

        def _compute_depth_map() -> dict[str, int]:
            # NB: `prepare_features` *mutates* the RDB it runs on — for a keyless
            # table it injects an `__index__` column into the frame (not the
            # schema), so the next `canonicalize_key_types` (inside
            # `compute_dfs_features`) rejects it. Run it on a shallow-frame copy,
            # never the cached RDB: the mutation only *adds* a column, so a
            # `copy(deep=False)` (shared column data, no duplication) suffices.
            return {
                f.get_name(): f.get_depth()
                for f in get_dfs_engine(cfg.engine, cfg).prepare_features(
                    _shallow_frame_copy(_rdb()),
                    _dfs_input(),
                    key_mappings,
                    task.time_col,
                    cfg,
                )
            }

        def _matrix() -> pd.DataFrame:
            def validate(frame: pd.DataFrame) -> None:
                if len(frame) != len(table.df):
                    raise ValueError(
                        f"DFS cache has {len(frame)} rows; expected {len(table.df)}"
                    )

            return cached_frame(
                cache,
                lambda: str(
                    _dfs_cache_key(
                        db, task, table, history_table, max_depth, run_identity
                    )
                ),
                _compute_matrix,
                validate=validate,
            )

        def _depth_map() -> dict[str, int]:
            return _cached_depth_map(
                cache,
                lambda: _dfs_depth_map_key(
                    db, task, history_table, max_depth, run_identity
                ),
                _compute_depth_map,
            )

        matrix = _CACHE.full_matrix(
            db, source_df, history_key_df, max_depth, cache, _matrix
        )
        depth_map = _CACHE.depth_map(db, history_key_df, max_depth, cache, _depth_map)

    if depth >= max_depth:
        sliced = matrix
    else:
        keep = [c for c in matrix.columns if depth_map.get(c, -1) <= depth]
        sliced = matrix.loc[:, keep]

    sliced = _temporal_diff(sliced, task.time_col)

    drop = {task.target_col}
    if keep_anchor_columns:
        if task.time_col and task.time_col in sliced.columns:
            sliced = _calendar_features(sliced, task.time_col)
    else:
        drop.add(task.entity_col)
        if task.time_col:
            drop.add(task.time_col)
    return type_columns(sliced, drop)
