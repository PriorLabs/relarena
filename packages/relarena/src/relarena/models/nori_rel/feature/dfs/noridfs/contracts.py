"""Backend-neutral contracts for native relational feature generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote

import pandas as pd

ColumnKind = Literal["numeric", "categorical", "datetime"]
PathDirection = Literal["bridge", "descendant", "parent"]
FeatureOperation = Literal[
    "anchor",
    "direct",
    "count",
    "last",
    "lookup",
    "max",
    "mean",
    "mean_age",
    "missing_fraction",
    "min",
    "mode",
    "oldest_age",
    "recency",
    "std",
    "sum",
]


@dataclass(frozen=True, order=True)
class ForeignKey:
    """One single-column child-to-parent relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str

    @property
    def identity(self) -> str:
        """Return a stable, role-aware relationship identity."""
        return (
            f"{self.child_table}.{self.child_column}->"
            f"{self.parent_table}.{self.parent_column}"
        )


@dataclass(frozen=True)
class DatabaseView:
    """In-memory relational frames and their structural metadata.

    The shape mirrors the public fields on ``Snapshot`` without importing the
    connector package. A future RelBench adapter can construct the same view.
    """

    tables: Mapping[str, pd.DataFrame]
    primary_keys: Mapping[str, str]
    foreign_keys: tuple[ForeignKey, ...] = ()
    time_columns: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        tables: Mapping[str, pd.DataFrame],
        primary_keys: Mapping[str, str | Sequence[str]],
        foreign_keys: Sequence[ForeignKey | tuple[str, str, str, str]] = (),
        time_columns: Mapping[str, str] | None = None,
    ) -> DatabaseView:
        """Build a view from Snapshot-like mappings and fail on composite keys."""
        keys: dict[str, str] = {}
        for table, value in primary_keys.items():
            if isinstance(value, str):
                keys[table] = value
                continue
            columns = tuple(value)
            if len(columns) != 1:
                raise ValueError(
                    f"native features require one primary-key column for {table!r}"
                )
            keys[table] = columns[0]
        relations = tuple(
            relation if isinstance(relation, ForeignKey) else ForeignKey(*relation)
            for relation in foreign_keys
        )
        view = cls(
            tables=dict(tables),
            primary_keys=keys,
            foreign_keys=tuple(sorted(set(relations))),
            time_columns=dict(time_columns or {}),
        )
        view.validate()
        return view

    def validate(self) -> None:
        """Validate only structural facts needed by the native executor."""
        for name, frame in self.tables.items():
            if not isinstance(name, str) or not name:
                raise ValueError("table names must be non-empty strings")
            if not all(isinstance(column, str) for column in frame.columns):
                raise TypeError(f"columns in {name!r} must be strings")
            if not frame.columns.is_unique:
                raise ValueError(f"columns in {name!r} must be unique")
        for table, column in self.primary_keys.items():
            self._require_column(table, column, "primary key")
        for table, column in self.time_columns.items():
            self._require_column(table, column, "time column")
        for relation in self.foreign_keys:
            self._require_column(
                relation.child_table, relation.child_column, "foreign key"
            )
            self._require_column(
                relation.parent_table, relation.parent_column, "referred key"
            )
            if self.primary_keys.get(relation.parent_table) != relation.parent_column:
                raise ValueError(
                    "native features require foreign keys to reference the "
                    f"parent primary key: {relation.identity}"
                )

    def _require_column(self, table: str, column: str, role: str) -> None:
        if table not in self.tables:
            raise ValueError(f"{role} references missing table {table!r}")
        if column not in self.tables[table]:
            raise ValueError(f"{role} references missing column {table}.{column}")

    @property
    def schema_digest(self) -> str:
        """Hash the order-independent schema used to validate transforms."""
        schema = {
            "tables": {
                name: sorted(
                    (str(column), str(frame[column].dtype)) for column in frame.columns
                )
                for name, frame in sorted(self.tables.items())
            },
            "primary_keys": sorted(self.primary_keys.items()),
            "foreign_keys": sorted(relation.identity for relation in self.foreign_keys),
            "time_columns": sorted(self.time_columns.items()),
        }
        encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2s(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class AnchorDefinition:
    """How prediction rows identify an entity and point-in-time cutoff."""

    entity_table: str
    entity_column: str
    cutoff_column: str | None
    horizon_ns: int | None = None

    @classmethod
    def create(
        cls,
        *,
        entity_table: str,
        entity_column: str,
        cutoff_column: str | None,
        horizon: str | pd.Timedelta | None = None,
    ) -> AnchorDefinition:
        """Normalize an optional prediction horizon to nanoseconds."""
        horizon_ns = None
        if horizon is not None:
            delta = pd.Timedelta(horizon)
            if delta <= pd.Timedelta(0):
                raise ValueError("horizon must be positive")
            horizon_ns = int(delta.value)
        if cutoff_column is None and horizon_ns is not None:
            raise ValueError("horizon requires a cutoff column")
        return cls(entity_table, entity_column, cutoff_column, horizon_ns)


@dataclass(frozen=True)
class NativeFeatureConfig:
    """Small, deterministic native feature-search budget."""

    max_depth: int = 2
    max_paths: int = 24
    max_features: int = 256
    max_direct_features: int = 64
    max_direct_categorical_cardinality: int | None = None
    max_categorical_cardinality: int = 32
    window_multipliers: tuple[int, ...] = (1, 4, 16)
    include_missing_in_modes: bool = False
    include_missing_fractions: bool = False

    def __post_init__(self) -> None:
        """Validate the bounded search configuration."""
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if self.max_paths < 1:
            raise ValueError("max_paths must be positive")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.max_direct_features < 0:
            raise ValueError("max_direct_features must be non-negative")
        if (
            self.max_direct_categorical_cardinality is not None
            and self.max_direct_categorical_cardinality < 1
        ):
            raise ValueError(
                "max_direct_categorical_cardinality must be positive when set"
            )
        if self.max_categorical_cardinality < 1:
            raise ValueError("max_categorical_cardinality must be positive")
        if any(value < 1 for value in self.window_multipliers):
            raise ValueError("window multipliers must be positive")
        if len(set(self.window_multipliers)) != len(self.window_multipliers):
            raise ValueError("window multipliers must be unique")


@dataclass(frozen=True)
class RelationPath:
    """One bounded descendant, parent, or association-to-parent path."""

    entity_table: str
    edges: tuple[ForeignKey, ...]
    direction: PathDirection = "descendant"

    def __post_init__(self) -> None:
        """Require a non-empty path whose edges match its direction."""
        if self.direction not in {"bridge", "descendant", "parent"}:
            raise ValueError(f"unknown path direction {self.direction!r}")
        if not self.edges:
            raise ValueError("relation paths require at least one edge")
        if self.is_bridge:
            if len(self.edges) < 2:
                raise ValueError("bridge paths require a descendant and a parent edge")
            first, *parents = self.edges
            if first.parent_table != self.entity_table:
                raise ValueError("bridge path must start at the entity table")
            current = first.child_table
            visited = {self.entity_table, current}
            for edge in parents:
                if edge.child_table != current:
                    raise ValueError("bridge path parent edges are not contiguous")
                current = edge.parent_table
                if current in visited:
                    raise ValueError("relation paths cannot revisit a table")
                visited.add(current)
            return
        current = self.entity_table
        visited = {current}
        for edge in self.edges:
            expected = (
                edge.parent_table
                if self.direction == "descendant"
                else edge.child_table
            )
            if expected != current:
                raise ValueError("relation path edges are not contiguous")
            current = (
                edge.child_table
                if self.direction == "descendant"
                else edge.parent_table
            )
            if current in visited:
                raise ValueError("relation paths cannot revisit a table")
            visited.add(current)

    @property
    def terminal_table(self) -> str:
        """Return the final table reached by the path."""
        edge = self.edges[-1]
        return edge.child_table if self.direction == "descendant" else edge.parent_table

    @property
    def is_parent(self) -> bool:
        """Return whether every edge is traversed child-to-parent."""
        return self.direction == "parent"

    @property
    def is_bridge(self) -> bool:
        """Return whether one descendant edge is followed by parent edges."""
        return self.direction == "bridge"

    @property
    def identity(self) -> str:
        """Return a canonical path identity."""
        if self.is_bridge:
            first, *parents = self.edges
            upward = "/".join(f"U:{edge.identity}" for edge in parents)
            return f"{self.entity_table}|D:{first.identity}|{upward}"
        if self.is_parent:
            steps = "/".join(edge.identity for edge in self.edges)
            return f"{self.entity_table}=>{steps}"
        steps = "<-".join(
            f"{edge.child_table}.{edge.child_column}" for edge in self.edges
        )
        return f"{self.entity_table}<-{steps}"


@dataclass(frozen=True)
class FeatureSpec:
    """One output column in a frozen native feature plan."""

    name: str
    operation: FeatureOperation
    kind: ColumnKind
    table: str
    column: str | None = None
    path: RelationPath | None = None
    window_ns: int | None = None

    @classmethod
    def anchor(
        cls,
        table: str,
        component: str,
        kind: ColumnKind,
    ) -> FeatureSpec:
        """Construct a feature known directly from each prediction anchor."""
        name = f"nr|anchor|{_escape(component)}"
        return cls(name, "anchor", kind, table, component)

    @classmethod
    def direct(cls, table: str, column: str, kind: ColumnKind) -> FeatureSpec:
        """Construct a direct entity-column feature."""
        name = f"nr|direct|{_escape(table)}.{_escape(column)}"
        return cls(name, "direct", kind, table, column)

    @classmethod
    def lookup(
        cls,
        path: RelationPath,
        column: str,
        kind: ColumnKind,
    ) -> FeatureSpec:
        """Construct one many-to-one parent lookup feature."""
        if not path.is_parent:
            raise ValueError("lookup features require a parent path")
        name = f"nr|path|{_escape(path.identity)}|{_escape(column)}|lookup|all"
        return cls(name, "lookup", kind, path.terminal_table, column, path)

    @classmethod
    def aggregate(
        cls,
        path: RelationPath,
        operation: FeatureOperation,
        *,
        column: str | None = None,
        kind: ColumnKind = "numeric",
        window_ns: int | None = None,
    ) -> FeatureSpec:
        """Construct a path aggregate with a canonical readable name."""
        source = "rows" if column is None else _escape(column)
        window = "all" if window_ns is None else f"{window_ns}ns"
        name = f"nr|path|{_escape(path.identity)}|{source}|{operation}|{window}"
        return cls(
            name,
            operation,
            kind,
            path.terminal_table,
            column,
            path,
            window_ns,
        )


@dataclass(frozen=True)
class FeaturePlan:
    """Fit-frozen paths and output columns used for every transform."""

    schema_digest: str
    anchors: AnchorDefinition
    config: NativeFeatureConfig
    paths: tuple[RelationPath, ...]
    features: tuple[FeatureSpec, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the exact fitted output schema."""
        return tuple(feature.name for feature in self.features)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """Return the fitted categorical subset."""
        return tuple(
            feature.name for feature in self.features if feature.kind == "categorical"
        )

    @property
    def digest(self) -> str:
        """Return a stable digest suitable for persistent cache keys."""
        payload = {
            "schema": self.schema_digest,
            "anchors": self.anchors.__dict__,
            "config": self.config.__dict__,
            "paths": [path.identity for path in self.paths],
            "features": [
                {
                    "name": feature.name,
                    "operation": feature.operation,
                    "kind": feature.kind,
                    "table": feature.table,
                    "column": feature.column,
                    "path": feature.path.identity if feature.path is not None else None,
                    "window_ns": feature.window_ns,
                }
                for feature in self.features
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2s(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class NativeFeatureMatrix:
    """One aligned native feature matrix and its categorical schema."""

    frame: pd.DataFrame
    categorical_columns: tuple[str, ...]
    plan_digest: str


def _escape(value: str) -> str:
    return quote(value, safe="")


__all__ = [
    "AnchorDefinition",
    "ColumnKind",
    "DatabaseView",
    "FeaturePlan",
    "FeatureSpec",
    "ForeignKey",
    "NativeFeatureConfig",
    "NativeFeatureMatrix",
    "PathDirection",
    "RelationPath",
]
