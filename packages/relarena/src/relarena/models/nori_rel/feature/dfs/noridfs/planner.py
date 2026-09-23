"""Deterministic, bounded planning for native relational features."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable

import numpy as np
import pandas as pd

from .contracts import (
    AnchorDefinition,
    ColumnKind,
    DatabaseView,
    FeaturePlan,
    FeatureSpec,
    ForeignKey,
    NativeFeatureConfig,
    PathDirection,
    RelationPath,
)


def build_feature_plan(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
    *,
    excluded_columns: dict[str, set[str]] | None = None,
) -> FeaturePlan:
    """Create a task-name-free plan from fitted schema information."""
    database.validate()
    if anchors.entity_table not in database.tables:
        raise ValueError(f"missing entity table {anchors.entity_table!r}")
    if anchors.entity_table not in database.primary_keys:
        raise ValueError("entity table requires a single-column primary key")
    excluded = excluded_columns or {}
    root = database.tables[anchors.entity_table]
    root_key = database.primary_keys[anchors.entity_table]
    root_foreign_keys = {
        relation.child_column
        for relation in database.foreign_keys
        if relation.child_table == anchors.entity_table
    }
    anchor_features = [
        FeatureSpec.anchor(anchors.entity_table, "entity_key", "categorical")
    ]
    if anchors.cutoff_column is not None:
        anchor_features.extend(
            FeatureSpec.anchor(anchors.entity_table, component, "numeric")
            for component in (
                "cutoff_epoch_days",
                "cutoff_year",
                "cutoff_month",
                "cutoff_day",
                "cutoff_day_of_week",
            )
        )
    direct: list[FeatureSpec] = []
    direct_cardinality: dict[str, int] = {}
    for column in sorted(root.columns):
        if (
            column == root_key
            or column in root_foreign_keys
            or column in excluded.get(anchors.entity_table, set())
        ):
            continue
        kind = _column_kind(root[column])
        if kind is not None:
            if (
                kind == "categorical"
                and config.max_direct_categorical_cardinality is not None
            ):
                cardinality = _bounded_cardinality(
                    root[column], config.max_direct_categorical_cardinality
                )
                if cardinality is None:
                    continue
                direct_cardinality[column] = cardinality
            direct.append(FeatureSpec.direct(anchors.entity_table, column, kind))
    direct.sort(
        key=lambda feature: (
            feature.kind == "categorical",
            direct_cardinality.get(feature.column or "", 0),
            feature.name,
        )
    )

    paths = _enumerate_paths(database, anchors, config)
    direct_share = (
        config.max_features if not paths else max(1, config.max_features // 3)
    )
    selected = anchor_features[: config.max_features]
    direct_limit = min(
        config.max_direct_features,
        direct_share,
        config.max_features - len(selected),
    )
    selected.extend(direct[:direct_limit])
    tiers = _candidate_tiers(database, anchors, config, paths, excluded)
    for tier in tiers:
        for feature in tier:
            if len(selected) >= config.max_features:
                break
            selected.append(feature)
        if len(selected) >= config.max_features:
            break
    if not selected:
        raise ValueError("native relational planning produced no features")
    names = [feature.name for feature in selected]
    if len(names) != len(set(names)):
        raise RuntimeError("native feature names are not unique")
    return FeaturePlan(
        schema_digest=database.schema_digest,
        anchors=anchors,
        config=config,
        paths=paths,
        features=tuple(selected),
    )


def _enumerate_paths(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
) -> tuple[RelationPath, ...]:
    incoming: dict[str, list[ForeignKey]] = defaultdict(list)
    outgoing: dict[str, list[ForeignKey]] = defaultdict(list)
    for relation in database.foreign_keys:
        incoming[relation.parent_table].append(relation)
        outgoing[relation.child_table].append(relation)
    for relations in (*incoming.values(), *outgoing.values()):
        relations.sort()

    candidates: dict[PathDirection, list[RelationPath]] = defaultdict(list)
    directions: tuple[tuple[PathDirection, dict[str, list[ForeignKey]]], ...] = (
        ("parent", outgoing),
        ("descendant", incoming),
    )
    for direction, adjacency in directions:
        queue: deque[tuple[str, tuple[ForeignKey, ...], frozenset[str]]] = deque(
            [(anchors.entity_table, (), frozenset({anchors.entity_table}))]
        )
        while queue and len(candidates[direction]) < config.max_paths:
            current, prefix, visited = queue.popleft()
            if len(prefix) >= config.max_depth:
                continue
            for relation in adjacency.get(current, []):
                next_table = (
                    relation.parent_table
                    if direction == "parent"
                    else relation.child_table
                )
                if next_table in visited:
                    continue
                edges = (*prefix, relation)
                candidate = RelationPath(anchors.entity_table, edges, direction)
                if candidate.is_parent or _path_has_usable_time(
                    database, candidate, anchors
                ):
                    candidates[direction].append(candidate)
                    if len(candidates[direction]) >= config.max_paths:
                        break
                if next_table in database.primary_keys:
                    queue.append((next_table, edges, visited | {next_table}))
    candidates["bridge"].extend(
        _enumerate_bridge_paths(database, anchors, config, incoming, outgoing)
    )
    ordered_families = [
        sorted(
            candidates[direction], key=lambda path: (len(path.edges), path.identity)
        )[: config.max_paths]
        for direction in ("parent", "descendant", "bridge")
    ]
    # Give every available family a deterministic share before taking a second
    # path from any one family. A wide schema cannot starve all bridge paths.
    ordered = [
        family[index]
        for index in range(max(map(len, ordered_families), default=0))
        for family in ordered_families
        if index < len(family)
    ]
    return tuple(ordered[: config.max_paths])


def _enumerate_bridge_paths(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
    incoming: dict[str, list[ForeignKey]],
    outgoing: dict[str, list[ForeignKey]],
) -> list[RelationPath]:
    """Enumerate one descendant association edge followed by parent edges."""
    if config.max_depth < 2:
        return []
    candidates: list[RelationPath] = []
    queue: deque[tuple[str, tuple[ForeignKey, ...], frozenset[str]]] = deque()
    for association_edge in incoming.get(anchors.entity_table, []):
        association = association_edge.child_table
        if (
            anchors.cutoff_column is not None
            and association not in database.time_columns
        ):
            continue
        queue.append(
            (
                association,
                (association_edge,),
                frozenset({anchors.entity_table, association}),
            )
        )
    while queue and len(candidates) < config.max_paths:
        current, prefix, visited = queue.popleft()
        if len(prefix) >= config.max_depth:
            continue
        for relation in outgoing.get(current, []):
            next_table = relation.parent_table
            if next_table in visited:
                continue
            edges = (*prefix, relation)
            candidate = RelationPath(anchors.entity_table, edges, "bridge")
            candidates.append(candidate)
            if len(candidates) >= config.max_paths:
                break
            if len(edges) < config.max_depth:
                queue.append((next_table, edges, visited | {next_table}))
    return candidates


def _path_has_usable_time(
    database: DatabaseView,
    path: RelationPath,
    anchors: AnchorDefinition,
) -> bool:
    if anchors.cutoff_column is None:
        return True
    if path.is_bridge:
        # A parent timestamp cannot establish when a timeless association
        # became visible. The first child is the bridge's occurrence clock.
        return path.edges[0].child_table in database.time_columns
    # A root creation time cannot establish when timeless child facts occurred.
    return any(edge.child_table in database.time_columns for edge in path.edges)


def _candidate_tiers(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
    paths: tuple[RelationPath, ...],
    excluded: dict[str, set[str]],
) -> list[list[FeatureSpec]]:
    parent_paths = tuple(path for path in paths if path.is_parent)
    aggregate_paths = tuple(
        path for path in paths if path.direction in {"bridge", "descendant"}
    )
    parent_columns: dict[RelationPath, tuple[list[str], list[str]]] = {}
    parent_kinds: dict[tuple[RelationPath, str], ColumnKind] = {}
    for path in parent_paths:
        table = path.terminal_table
        frame = database.tables[table]
        keys = {database.primary_keys.get(table)} | {
            relation.child_column
            for relation in database.foreign_keys
            if relation.child_table == table
        }
        keys.discard(None)
        ignored = keys | excluded.get(table, set())
        primary: list[str] = []
        categorical: list[str] = []
        for column in sorted(frame.columns):
            if column in ignored:
                continue
            kind = _column_kind(frame[column])
            if kind is None:
                continue
            parent_kinds[(path, column)] = kind
            destination = categorical if kind == "categorical" else primary
            destination.append(column)
        parent_columns[path] = (primary, categorical)

    aggregate_columns: dict[RelationPath, tuple[list[str], list[str]]] = {}
    for path in aggregate_paths:
        table = path.terminal_table
        frame = database.tables[table]
        keys = {database.primary_keys.get(table)} | {
            relation.child_column
            for relation in database.foreign_keys
            if relation.child_table == table
        }
        keys.discard(None)
        ignored = keys | {database.time_columns.get(table)} | excluded.get(table, set())
        ignored.discard(None)
        numeric: list[str] = []
        categorical: list[str] = []
        for column in sorted(frame.columns):
            if column in ignored:
                continue
            kind = _column_kind(frame[column])
            if kind == "numeric":
                numeric.append(column)
            elif (
                kind == "categorical"
                and _bounded_cardinality(
                    frame[column], config.max_categorical_cardinality
                )
                is not None
            ):
                categorical.append(column)
        aggregate_columns[path] = (numeric, categorical)

    tiers: list[list[FeatureSpec]] = [
        [FeatureSpec.aggregate(path, "count") for path in aggregate_paths]
    ]
    for group in (0, 1):
        tiers.append(
            _round_robin_columns(
                parent_paths,
                parent_columns,
                group,
                lambda path, column: FeatureSpec.lookup(
                    path,
                    column,
                    parent_kinds[(path, column)],
                ),
            )
        )
    if anchors.cutoff_column is not None:
        tiers.append(
            [FeatureSpec.aggregate(path, "recency") for path in aggregate_paths]
        )
        tiers.append(
            [FeatureSpec.aggregate(path, "mean_age") for path in aggregate_paths]
        )
        tiers.append(
            [FeatureSpec.aggregate(path, "oldest_age") for path in aggregate_paths]
        )

    # Spend the compact budget on stable distribution summaries first. Latest
    # values and sums remain available to the richer arms below.
    for operation in ("mean", "std", "min", "max"):
        tiers.append(
            _round_robin_columns(
                aggregate_paths,
                aggregate_columns,
                0,
                lambda path, column, op=operation: FeatureSpec.aggregate(
                    path, op, column=column
                ),
            )
        )

    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            1,
            lambda path, column: FeatureSpec.aggregate(
                path, "mode", column=column, kind="categorical"
            ),
        )
    )
    if config.include_missing_fractions:
        tiers.append(
            _round_robin_columns(
                aggregate_paths,
                aggregate_columns,
                1,
                lambda path, column: FeatureSpec.aggregate(
                    path,
                    "missing_fraction",
                    column=column,
                ),
            )
        )

    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            0,
            lambda path, column: FeatureSpec.aggregate(path, "last", column=column),
        )
    )
    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            1,
            lambda path, column: FeatureSpec.aggregate(
                path, "last", column=column, kind="categorical"
            ),
        )
    )

    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            0,
            lambda path, column: FeatureSpec.aggregate(path, "sum", column=column),
        )
    )
    if anchors.horizon_ns is not None:
        for multiplier in config.window_multipliers:
            window = anchors.horizon_ns * multiplier
            tiers.append(
                [
                    FeatureSpec.aggregate(path, "count", window_ns=window)
                    for path in aggregate_paths
                ]
            )
            for operation in ("mean", "std", "sum"):
                tiers.append(
                    _round_robin_columns(
                        aggregate_paths,
                        aggregate_columns,
                        0,
                        lambda path, column, op=operation, win=window: (
                            FeatureSpec.aggregate(
                                path,
                                op,
                                column=column,
                                window_ns=win,
                            )
                        ),
                    )
                )
    return tiers


def _round_robin_columns(
    paths: tuple[RelationPath, ...],
    columns: dict[RelationPath, tuple[list[str], list[str]]],
    group: int,
    build: Callable[[RelationPath, str], FeatureSpec],
) -> list[FeatureSpec]:
    """Interleave columns across paths so one wide table cannot take the budget."""
    width = max((len(columns[path][group]) for path in paths), default=0)
    return [
        build(path, columns[path][group][position])
        for position in range(width)
        for path in paths
        if position < len(columns[path][group])
    ]


def _column_kind(series: pd.Series) -> ColumnKind | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Object columns can mix scalar and nested values while retaining the same
    # dtype. Inspect every present value so row order cannot change the plan.
    for value in series:
        if value is None or (np.ndim(value) == 0 and pd.isna(value)):
            continue
        if not pd.api.types.is_scalar(value):
            return None
    return "categorical"


def _bounded_cardinality(series: pd.Series, limit: int) -> int | None:
    """Return bounded cardinality without materializing every distinct value."""
    values: set[object] = set()
    for value in series:
        if value is None or bool(pd.isna(value)):
            continue
        normalized = value.item() if isinstance(value, np.generic) else value
        values.add(normalized)
        if len(values) > limit:
            return None
    return len(values)


__all__ = ["build_feature_plan"]
