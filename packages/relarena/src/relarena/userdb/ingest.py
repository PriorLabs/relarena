"""Ingest a user-described relational database into RelBench objects.

A user points at their tables (parquet/CSV) and declares the schema — primary
key, time column and foreign-key links per table — via `DatabaseSpec`.
`build_dataset` turns that into a reindexed RelBench `UserDataset`
(with the val/test split timestamps) that flows through
`RelBenchDatasetTask.from_objects` exactly like a native RelBench dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from relbench.base import Database, Dataset, Table

from relarena.userdb._schema import load_schema, validate

#: JSON Schema for a database YAML; the single source of truth for its shape.
_DB_SCHEMA = load_schema("database.schema.json")


@dataclass(frozen=True)
class TableSource:
    """One table's data file plus its schema (primary key, time col, foreign keys).

    `columns` optionally restricts the table to a subset of its columns (the rest
    are dropped on ingest) - handy for pointing at a raw file while curating which
    columns become model features. It must include the pkey, time col and fkeys.
    """

    path: str
    pkey: str | None = None
    time_col: str | None = None
    fkeys: dict[str, str] = field(default_factory=dict)
    columns: list[str] | None = None


@dataclass(frozen=True)
class DatabaseSpec:
    """A relational database: its named table sources and their schema.

    Carries no val/test timestamps — those define an evaluation split, not the
    data, and enter at `build_dataset` / `UserDataset` (mirroring
    RelBench's `Database` vs `Dataset` split).
    """

    tables: dict[str, TableSource]

    @classmethod
    def from_yaml(cls, path: str, *, data_dir: str | None = None) -> DatabaseSpec:
        """Load a database schema from a YAML file (table name -> its schema).

        Each table's data file is its `path` if given; relative paths and the
        default `<name>.parquet` resolve against `data_dir`, or the YAML's own
        directory if `data_dir` is omitted - so the schema stays portable and
        `materialize_relbench` can supply the files.
        """
        raw = yaml.safe_load(Path(path).read_text())
        validate(raw, _DB_SCHEMA, kind="database")
        base = Path(data_dir) if data_dir is not None else Path(path).parent
        tables = {}
        for name, entry in raw.items():
            entry = entry or {}
            rel = entry.get("path", f"{name}.parquet")
            full = rel if Path(rel).is_absolute() else str(base / rel)
            tables[name] = TableSource(
                path=full,
                pkey=entry.get("pkey"),
                time_col=entry.get("time_col"),
                fkeys=entry.get("fkeys") or {},
                columns=entry.get("columns"),
            )
        return cls(tables=tables)


def _read_table(path: str) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _reindex_stable(db: Database) -> dict[str, pd.Series]:
    """Reindex pkeys and fkeys to `0..n-1`, sorting temporal tables stably by time.

    Mirrors RelBench's `reindex_pkeys_and_fkeys` but sorts `time_col` with a stable
    sort and returns each pkey table's `original id -> index` map. RelBench sorts
    non-stably, so a table with tied timestamps gets a tie order that depends on
    input row order - ingesting the same data reordered, or re-ingesting RelBench's
    own already-time-sorted tables, would reassign entity ids. A stable sort keeps
    the input order among ties, making the assignment deterministic - and, for
    re-ingested RelBench tables, byte-for-byte identical to native ids, so models
    run through this interface stay reproducible against native RelBench.
    """
    index_maps: dict[str, pd.Series] = {}
    for name, table in db.table_dict.items():
        if table.pkey_col is None:
            continue
        if table.time_col is not None:
            table.df = table.df.sort_values(table.time_col, kind="stable").reset_index(
                drop=True
            )
        ser = table.df[table.pkey_col]
        if ser.nunique() != len(ser):
            raise ValueError(
                f"Primary key {table.pkey_col!r} of table {name!r} has duplicates."
            )
        ids = pd.RangeIndex(len(ser)).astype("Int64")
        index_maps[name] = pd.Series(index=ser, data=ids, name="index")
        table.df[table.pkey_col] = ids
    for table in db.table_dict.values():
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            merged = pd.merge(
                table.df[fkey_col],
                index_maps[pkey_table],
                how="left",
                left_on=fkey_col,
                right_index=True,
            )
            table.df[fkey_col] = merged["index"]
    return index_maps


class _StableDatabase(Database):
    """A `Database` whose reindex sorts `time_col` stably (see `_reindex_stable`).

    RelBench's `Dataset.get_db` reindexes the `make_db` result with the base
    `reindex_pkeys_and_fkeys`, which is non-stable; on tie-heavy temporal tables
    that re-shuffles entity ids and diverges from native RelBench - undoing the
    stable reindex `_build_database` already did. Using a stable reindex there too
    keeps `get_db`'s output byte-for-byte identical to native (and idempotent on the
    already-reindexed db `_build_database` produces).
    """

    def reindex_pkeys_and_fkeys(self) -> None:
        """Reindex stably in place, matching `_build_database`'s ingest reindex."""
        _reindex_stable(self)


def _build_database(spec: DatabaseSpec) -> tuple[Database, dict[str, pd.Series]]:
    """Build the reindexed RelBench `Database` and capture each table's id map.

    Reindexing maps each table's primary key (and the foreign keys pointing at it)
    to the consecutive `0..n-1` integers RelBench requires, so arbitrary user ids
    (strings, gaps) are accepted. `_reindex_stable` also returns each pkey table's
    `original id -> 0..n-1` map, so callers can translate between the user's real
    ids and the internal indices.
    """
    table_dict: dict[str, Table] = {}
    for name, src in spec.tables.items():
        df = _read_table(src.path)
        if src.columns is not None:
            keys = {c for c in (src.pkey, src.time_col, *src.fkeys) if c is not None}
            if keys - set(src.columns):
                raise ValueError(
                    f"Table {name!r}: pkey/time_col/fkey column(s) "
                    f"{sorted(keys - set(src.columns))} must be listed in `columns`."
                )
            if missing := [c for c in src.columns if c not in df.columns]:
                raise ValueError(
                    f"Table {name!r}: `columns` not found in the data: {missing}."
                )
            df = df[src.columns]
        if src.time_col is not None:
            # Parse to datetime so censoring/sorting compare timestamps, not strings
            # (CSV reads them as strings; parquet is already typed).
            df[src.time_col] = pd.to_datetime(df[src.time_col])
        table_dict[name] = Table(
            df=df,
            fkey_col_to_pkey_table=dict(src.fkeys),
            pkey_col=src.pkey,
            time_col=src.time_col,
        )
    db = _StableDatabase(table_dict)
    index_maps = _reindex_stable(db)
    pkey_maps = {name: s.astype("int64") for name, s in index_maps.items()}
    return db, pkey_maps


class UserDataset(Dataset):
    """A RelBench `Dataset` backed by a prebuilt `Database` + split timestamps."""

    def __init__(
        self,
        db: Database,
        *,
        val_timestamp: pd.Timestamp,
        test_timestamp: pd.Timestamp,
        pkey_maps: dict[str, pd.Series] | None = None,
    ) -> None:
        """Store the database, split timestamps, and per-table original-id maps."""
        super().__init__(cache_dir=None)
        self._db = db
        self.val_timestamp = pd.Timestamp(val_timestamp)
        self.test_timestamp = pd.Timestamp(test_timestamp)
        #: table name -> Series mapping each table's original pkey to its reindexed
        #: `0..n-1` id, so callers can translate user ids to/from internal indices.
        self.pkey_maps = pkey_maps or {}

    def make_db(self) -> Database:
        """Return a fresh copy of the prebuilt database.

        RelBench's `get_db` reindexes, censors and dangling-FK-scrubs the made DB
        *in place*, and calls `make_db` separately for the full and test-censored
        variants. Returning a fresh copy (as RelBench's own `make_db` does) keeps
        those mutations from leaking into `self._db` or across `get_db` calls.

        Returns a `_StableDatabase` so `get_db`'s reindex is stable (matching the
        ingest reindex), not RelBench's non-stable one.
        """
        return _StableDatabase(
            {
                name: Table(
                    df=table.df.copy(),
                    fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                    pkey_col=table.pkey_col,
                    time_col=table.time_col,
                )
                for name, table in self._db.table_dict.items()
            }
        )


def build_dataset(
    spec: DatabaseSpec,
    *,
    val_timestamp: pd.Timestamp,
    test_timestamp: pd.Timestamp,
) -> UserDataset:
    """Build a `UserDataset` from the manifest plus the val/test split timestamps."""
    db, pkey_maps = _build_database(spec)
    return UserDataset(
        db,
        val_timestamp=val_timestamp,
        test_timestamp=test_timestamp,
        pkey_maps=pkey_maps,
    )
