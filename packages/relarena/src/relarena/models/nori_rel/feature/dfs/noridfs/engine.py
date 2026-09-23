"""Native point-in-time relational feature execution."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    FeatureSpec,
    NativeFeatureConfig,
    NativeFeatureMatrix,
    RelationPath,
)
from .planner import build_feature_plan

_ROW = "__nr_row__"
_ENTITY = "__nr_entity__"
_NODE = "__nr_node__"
_JOIN = "__nr_join__"
_NEXT = "__nr_next__"
_TIME = "__nr_time__"
_STEP_TIME = "__nr_step_time__"
_AT = "__nr_at__"
_TIE = "__nr_tie__"
_DAY_NS = float(pd.Timedelta(days=1).value)
_MISSING_CATEGORY = "__nori_rel_missing__"
_CATEGORY_ESCAPE = "__nori_rel_escape__:"


class NativeRelationalFeaturizer:
    """Execute one immutable, fit-frozen relational feature plan."""

    def __init__(self, plan: FeaturePlan) -> None:
        """Store an immutable fitted plan."""
        self.plan = plan

    @classmethod
    def fit(
        cls,
        database: DatabaseView,
        anchors: AnchorDefinition,
        config: NativeFeatureConfig | None = None,
        *,
        excluded_columns: Mapping[str, set[str]] | None = None,
    ) -> NativeRelationalFeaturizer:
        """Freeze paths, operations, windows, and output order from fit schema."""
        excluded = {
            table: set(columns) for table, columns in (excluded_columns or {}).items()
        }
        plan = build_feature_plan(
            database,
            anchors,
            config or NativeFeatureConfig(),
            excluded_columns=excluded,
        )
        return cls(plan)

    def transform(
        self, database: DatabaseView, anchor_frame: pd.DataFrame
    ) -> NativeFeatureMatrix:
        """Build one row per anchor using only rows strictly before its cutoff."""
        database.validate()
        if database.schema_digest != self.plan.schema_digest:
            raise ValueError("database schema does not match the fitted feature plan")
        anchors = _prepare_anchors(anchor_frame, self.plan.anchors)
        values: dict[str, pd.Series | np.ndarray] = {}
        anchor_specs = [
            feature for feature in self.plan.features if feature.operation == "anchor"
        ]
        values.update(_anchor_features(anchors, anchor_specs))
        direct = [
            feature for feature in self.plan.features if feature.operation == "direct"
        ]
        values.update(_direct_features(database, anchors, self.plan.anchors, direct))

        by_path: dict[RelationPath, list[FeatureSpec]] = defaultdict(list)
        for feature in self.plan.features:
            if feature.path is not None:
                by_path[feature.path].append(feature)
        for path in self.plan.paths:
            specs = by_path.get(path)
            if specs:
                values.update(
                    _path_features(
                        database,
                        anchors,
                        path,
                        specs,
                        self.plan.config.include_missing_in_modes,
                    )
                )

        missing = [column for column in self.plan.columns if column not in values]
        if missing:
            raise RuntimeError(
                f"native executor omitted planned features: {missing[:3]}"
            )
        frame = pd.DataFrame(values).reindex(columns=list(self.plan.columns))
        if len(frame) != len(anchor_frame):
            raise RuntimeError("native feature row count changed during execution")
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_columns] = frame[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        frame.reset_index(drop=True, inplace=True)
        return NativeFeatureMatrix(
            frame=frame,
            categorical_columns=self.plan.categorical_columns,
            plan_digest=self.plan.digest,
        )


def _prepare_anchors(frame: pd.DataFrame, definition: AnchorDefinition) -> pd.DataFrame:
    required = [definition.entity_column]
    if definition.cutoff_column is not None:
        required.append(definition.cutoff_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"anchor columns are missing: {missing}")
    anchors = pd.DataFrame(
        {
            _ROW: np.arange(len(frame), dtype=np.int64),
            _ENTITY: frame[definition.entity_column].reset_index(drop=True),
        }
    )
    if anchors[_ENTITY].isna().any():
        raise ValueError("anchor entity keys must be present")
    if definition.cutoff_column is not None:
        anchors[_AT] = _time_ns(frame[definition.cutoff_column])
        if anchors[_AT].isna().any():
            raise ValueError("anchor cutoffs must be valid timestamps")
    return anchors


def _anchor_features(
    anchors: pd.DataFrame,
    specs: list[FeatureSpec],
) -> dict[str, pd.Series | np.ndarray]:
    """Materialize fitted entity-key and compact cutoff features."""
    if not specs:
        return {}
    timestamps = (
        pd.to_datetime(anchors[_AT], unit="ns", utc=True) if _AT in anchors else None
    )
    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.column == "entity_key":
            output[spec.name] = anchors[_ENTITY].astype(object).reset_index(drop=True)
        elif timestamps is None:
            raise RuntimeError(f"anchor feature {spec.column!r} requires a cutoff")
        elif spec.column == "cutoff_epoch_days":
            output[spec.name] = anchors[_AT].to_numpy(dtype=float) / _DAY_NS
        elif spec.column == "cutoff_year":
            output[spec.name] = timestamps.dt.year.to_numpy(dtype=float)
        elif spec.column == "cutoff_month":
            output[spec.name] = timestamps.dt.month.to_numpy(dtype=float)
        elif spec.column == "cutoff_day":
            output[spec.name] = timestamps.dt.day.to_numpy(dtype=float)
        elif spec.column == "cutoff_day_of_week":
            output[spec.name] = timestamps.dt.dayofweek.to_numpy(dtype=float)
        else:
            raise RuntimeError(f"unknown anchor feature {spec.column!r}")
    return output


def _direct_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    definition: AnchorDefinition,
    specs: list[FeatureSpec],
) -> dict[str, pd.Series]:
    if not specs:
        return {}
    root = database.tables[definition.entity_table]
    primary_key = database.primary_keys[definition.entity_table]
    if root[primary_key].isna().any():
        raise ValueError(
            f"primary key {definition.entity_table}.{primary_key} is missing"
        )
    anchor_keys, root_keys = _aligned_keys(anchors[_ENTITY], root[primary_key])
    if root_keys.dropna().duplicated().any():
        raise ValueError(
            f"primary key {definition.entity_table}.{primary_key} is not unique"
        )

    left = anchors.assign(**{_ENTITY: anchor_keys})
    right = pd.DataFrame({_NODE: root_keys})
    aliases: dict[str, str] = {}
    for index, column in enumerate(sorted({spec.column for spec in specs})):
        if column is None:
            continue
        alias = f"__nr_direct_{index}__"
        aliases[column] = alias
        right[alias] = root[column].reset_index(drop=True)
    root_time = database.time_columns.get(definition.entity_table)
    if root_time is not None:
        right[_TIME] = _time_ns(root[root_time])
    merged = left.merge(
        right,
        how="left",
        left_on=_ENTITY,
        right_on=_NODE,
        sort=False,
        validate="many_to_one",
    ).sort_values(_ROW, kind="mergesort")
    visible = np.ones(len(merged), dtype=bool)
    if definition.cutoff_column is not None and root_time is not None:
        visible = merged[_TIME].to_numpy(dtype=float) < merged[_AT].to_numpy(
            dtype=float
        )

    output: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.column is None:
            raise RuntimeError("direct feature has no source column")
        source = merged[aliases[spec.column]]
        if spec.kind == "datetime":
            timestamp = _time_ns(source)
            if definition.cutoff_column is None:
                values = timestamp / _DAY_NS
            else:
                values = (merged[_AT].to_numpy(dtype=float) - timestamp) / _DAY_NS
            values[~visible] = np.nan
            output[spec.name] = pd.Series(values, dtype=float)
        elif spec.kind == "numeric":
            values = pd.to_numeric(source, errors="coerce").astype(float)
            output[spec.name] = values.where(visible).reset_index(drop=True)
        else:
            values = source.astype(object).where(visible, None)
            output[spec.name] = values.reset_index(drop=True)
    return output


def _path_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    path: RelationPath,
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    columns = sorted({spec.column for spec in specs if spec.column is not None})
    if path.is_parent:
        return _parent_path_features(database, anchors, path, columns, specs)
    events, sources = (
        _materialize_bridge_path(database, path, columns)
        if path.is_bridge
        else _materialize_path(database, path, columns)
    )
    if _AT not in anchors:
        return _static_path_features(
            anchors,
            events,
            sources,
            specs,
            include_missing_in_modes,
        )
    return _temporal_path_features(
        anchors,
        events,
        sources,
        specs,
        include_missing_in_modes,
    )


def _parent_path_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    path: RelationPath,
    columns: list[str],
    specs: list[FeatureSpec],
) -> dict[str, pd.Series]:
    """Look up one ancestor row per entity with point-in-time visibility."""
    rows, sources, timed = _materialize_parent_path(database, path, columns)
    anchor_keys, row_keys = _aligned_keys(anchors[_ENTITY], rows[_ENTITY])
    left = anchors.assign(**{_ENTITY: anchor_keys})
    right = rows.assign(**{_ENTITY: row_keys})
    if right[_ENTITY].dropna().duplicated().any():
        raise RuntimeError("parent path produced more than one row per entity")
    merged = left.merge(
        right,
        how="left",
        on=_ENTITY,
        sort=False,
        validate="many_to_one",
    ).sort_values(_ROW, kind="mergesort")
    visible = np.ones(len(merged), dtype=bool)
    if timed and _AT in anchors:
        event_time = merged[_TIME].to_numpy(dtype=float)
        visible = np.isfinite(event_time) & (
            event_time < merged[_AT].to_numpy(dtype=float)
        )

    output: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.operation != "lookup" or spec.column is None:
            raise RuntimeError("parent paths only support lookup features")
        source = merged[sources[spec.column]]
        if spec.kind == "datetime":
            timestamp = _time_ns(source)
            if _AT in anchors:
                values = (merged[_AT].to_numpy(dtype=float) - timestamp) / _DAY_NS
            else:
                values = timestamp / _DAY_NS
            values[~visible] = np.nan
            output[spec.name] = pd.Series(values, dtype=float)
        elif spec.kind == "numeric":
            values = pd.to_numeric(source, errors="coerce").astype(float)
            output[spec.name] = values.where(visible).reset_index(drop=True)
        else:
            values = source.astype(object).where(visible, None)
            output[spec.name] = values.reset_index(drop=True)
    return output


def _materialize_parent_path(
    database: DatabaseView,
    path: RelationPath,
    source_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str], bool]:
    """Follow a many-to-one parent chain without changing root cardinality."""
    root_keys = _primary_key(database, path.entity_table)
    mapping = pd.DataFrame(
        {
            _ENTITY: root_keys,
            _NODE: root_keys,
            _TIME: np.full(len(root_keys), np.nan),
        }
    )
    timed = False
    current = path.entity_table
    for relation in path.edges:
        if relation.child_table != current:
            raise RuntimeError("parent relation path is not contiguous")
        frame = database.tables[current]
        current_keys = _primary_key(database, current)
        mapping_keys, current_keys = _aligned_keys(mapping[_NODE], current_keys)
        mapping = mapping.assign(**{_NODE: mapping_keys})
        payload = pd.DataFrame(
            {
                _JOIN: current_keys,
                _NEXT: frame[relation.child_column].reset_index(drop=True),
            }
        )
        time_column = database.time_columns.get(current)
        payload[_STEP_TIME] = (
            _time_ns(frame[time_column])
            if time_column is not None
            else np.full(len(frame), np.nan)
        )
        step_timed = time_column is not None
        mapping = mapping.merge(
            payload,
            how="left",
            left_on=_NODE,
            right_on=_JOIN,
            sort=False,
            validate="many_to_one",
        )
        mapping[_TIME] = _compose_time(
            mapping[_TIME].to_numpy(dtype=float),
            mapping[_STEP_TIME].to_numpy(dtype=float),
            accumulated_timed=timed,
            step_timed=step_timed,
        )
        timed = timed or step_timed
        mapping = mapping[[_ENTITY, _NEXT, _TIME]].rename(columns={_NEXT: _NODE})
        current = relation.parent_table

    terminal = database.tables[current]
    terminal_keys = _primary_key(database, current)
    mapping_keys, terminal_keys = _aligned_keys(mapping[_NODE], terminal_keys)
    mapping = mapping.assign(**{_NODE: mapping_keys})
    payload = pd.DataFrame({_JOIN: terminal_keys})
    time_column = database.time_columns.get(current)
    payload[_STEP_TIME] = (
        _time_ns(terminal[time_column])
        if time_column is not None
        else np.full(len(terminal), np.nan)
    )
    step_timed = time_column is not None
    sources: dict[str, str] = {}
    for index, column in enumerate(source_columns):
        alias = f"__nr_source_{index}__"
        sources[column] = alias
        payload[alias] = terminal[column].reset_index(drop=True)
    mapping = mapping.merge(
        payload,
        how="left",
        left_on=_NODE,
        right_on=_JOIN,
        sort=False,
        validate="many_to_one",
    )
    mapping[_TIME] = _compose_time(
        mapping[_TIME].to_numpy(dtype=float),
        mapping[_STEP_TIME].to_numpy(dtype=float),
        accumulated_timed=timed,
        step_timed=step_timed,
    )
    timed = timed or step_timed
    columns = [_ENTITY, _TIME, *sources.values()]
    result = mapping[columns].reset_index(drop=True)
    if len(result) != len(root_keys):
        raise RuntimeError("parent path changed root cardinality")
    return result, sources, timed


def _primary_key(database: DatabaseView, table: str) -> pd.Series:
    """Return one validated primary-key vector."""
    column = database.primary_keys[table]
    values = database.tables[table][column].reset_index(drop=True)
    if values.isna().any():
        raise ValueError(f"primary key {table}.{column} is missing")
    if values.duplicated().any():
        raise ValueError(f"primary key {table}.{column} is not unique")
    return values


def _materialize_bridge_path(
    database: DatabaseView,
    path: RelationPath,
    source_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Preserve association occurrences while following a many-to-one tail."""
    if not path.is_bridge:
        raise RuntimeError("bridge materialization requires a bridge path")
    association_edge, *parent_edges = path.edges
    root = database.tables[path.entity_table]
    root_key = database.primary_keys[path.entity_table]
    root_keys = _primary_key(database, path.entity_table)
    root_time = database.time_columns.get(path.entity_table)
    mapping = pd.DataFrame(
        {
            _NODE: root_keys,
            _ENTITY: root_keys,
            _TIME: (
                _time_ns(root[root_time])
                if root_time is not None
                else np.full(len(root), np.nan)
            ),
        }
    )

    association = database.tables[association_edge.child_table]
    association_keys, aligned_roots = _aligned_keys(
        association[association_edge.child_column], mapping[_NODE]
    )
    mapping = mapping.assign(**{_NODE: aligned_roots})
    association_time = database.time_columns.get(association_edge.child_table)
    first_parent = parent_edges[0]
    payload = pd.DataFrame(
        {
            _JOIN: association_keys,
            _NEXT: association[first_parent.child_column].reset_index(drop=True),
            _STEP_TIME: (
                _time_ns(association[association_time])
                if association_time is not None
                else np.full(len(association), np.nan)
            ),
        }
    )
    association_key = database.primary_keys.get(association_edge.child_table)
    if association_key is not None:
        tie_values: pd.Series | pd.DataFrame = _primary_key(
            database, association_edge.child_table
        )
    else:
        tie_columns = list(
            dict.fromkeys(
                [
                    association_edge.child_column,
                    first_parent.child_column,
                    *([association_time] if association_time is not None else []),
                ]
            )
        )
        tie_values = association.reindex(columns=tie_columns)
    payload[_TIE] = pd.util.hash_pandas_object(
        tie_values,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    joined = payload[payload[_JOIN].notna()].merge(
        mapping[mapping[_NODE].notna()],
        how="inner",
        left_on=_JOIN,
        right_on=_NODE,
        sort=False,
        validate="many_to_one",
    )
    joined[_TIME] = _compose_time(
        joined[_TIME].to_numpy(dtype=float),
        joined[_STEP_TIME].to_numpy(dtype=float),
        accumulated_timed=root_time is not None,
        step_timed=association_time is not None,
    )
    mapping = joined[[_ENTITY, _NEXT, _TIME, _TIE]].rename(columns={_NEXT: _NODE})
    accumulated_timed = root_time is not None or association_time is not None
    current = association_edge.child_table
    sources: dict[str, str] = {}

    for index, relation in enumerate(parent_edges):
        if relation.child_table != current:
            raise RuntimeError("bridge parent tail is not contiguous")
        parent = database.tables[relation.parent_table]
        parent_keys = _primary_key(database, relation.parent_table)
        mapping_keys, parent_keys = _aligned_keys(mapping[_NODE], parent_keys)
        mapping = mapping.assign(**{_NODE: mapping_keys})
        parent_time = database.time_columns.get(relation.parent_table)
        parent_payload = pd.DataFrame(
            {
                _JOIN: parent_keys,
                _STEP_TIME: (
                    _time_ns(parent[parent_time])
                    if parent_time is not None
                    else np.full(len(parent), np.nan)
                ),
            }
        )
        terminal = index == len(parent_edges) - 1
        if terminal:
            for position, column in enumerate(source_columns):
                alias = f"__nr_source_{position}__"
                sources[column] = alias
                parent_payload[alias] = parent[column].reset_index(drop=True)
        else:
            next_relation = parent_edges[index + 1]
            parent_payload[_NEXT] = parent[next_relation.child_column].reset_index(
                drop=True
            )

        before = len(mapping)
        joined = mapping[mapping[_NODE].notna()].merge(
            parent_payload,
            how="inner",
            left_on=_NODE,
            right_on=_JOIN,
            sort=False,
            validate="many_to_one",
        )
        if len(joined) > before:
            raise RuntimeError("bridge parent join multiplied association rows")
        joined[_TIME] = _compose_time(
            joined[_TIME].to_numpy(dtype=float),
            joined[_STEP_TIME].to_numpy(dtype=float),
            accumulated_timed=accumulated_timed,
            step_timed=parent_time is not None,
        )
        accumulated_timed = accumulated_timed or parent_time is not None
        current = relation.parent_table
        if terminal:
            columns = [_ENTITY, _TIME, _TIE, *sources.values()]
            return joined[columns].reset_index(drop=True), sources
        mapping = joined[[_ENTITY, _NEXT, _TIME, _TIE]].rename(columns={_NEXT: _NODE})
    raise RuntimeError(f"bridge from {path.entity_table}.{root_key} has no parent tail")


def _materialize_path(
    database: DatabaseView,
    path: RelationPath,
    source_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if path.direction != "descendant":
        raise RuntimeError("descendant materialization requires a descendant path")
    root = database.tables[path.entity_table]
    root_key = database.primary_keys[path.entity_table]
    keys = root[root_key].reset_index(drop=True)
    if keys.isna().any():
        raise ValueError(f"primary key {path.entity_table}.{root_key} is missing")
    if keys.dropna().duplicated().any():
        raise ValueError(f"primary key {path.entity_table}.{root_key} is not unique")
    mapping = pd.DataFrame({_NODE: keys, _ENTITY: keys})
    root_time = database.time_columns.get(path.entity_table)
    mapping[_TIME] = (
        _time_ns(root[root_time])
        if root_time is not None
        else np.full(len(root), np.nan)
    )
    accumulated_timed = root_time is not None
    current = path.entity_table
    sources: dict[str, str] = {}
    for depth, relation in enumerate(path.edges):
        if relation.parent_table != current:
            raise RuntimeError("relation path is not contiguous")
        child = database.tables[relation.child_table]
        child_keys, parent_keys = _aligned_keys(
            child[relation.child_column], mapping[_NODE]
        )
        mapping = mapping.assign(**{_NODE: parent_keys})
        payload = pd.DataFrame(
            {
                _JOIN: child_keys,
            }
        )
        child_time = database.time_columns.get(relation.child_table)
        payload[_TIME] = (
            _time_ns(child[child_time])
            if child_time is not None
            else np.full(len(child), np.nan)
        )
        terminal = depth == len(path.edges) - 1
        if terminal:
            terminal_key = database.primary_keys.get(relation.child_table)
            if terminal_key is not None:
                tie_values = child[terminal_key]
            else:
                # Excluded and unsupported fields must not affect a planned
                # feature. Use only structural, temporal, and planned values.
                tie_columns = list(
                    dict.fromkeys(
                        [
                            relation.child_column,
                            *([child_time] if child_time is not None else []),
                            *source_columns,
                        ]
                    )
                )
                tie_values = child.reindex(columns=tie_columns)
            payload[_TIE] = pd.util.hash_pandas_object(
                tie_values,
                index=False,
                categorize=True,
            ).to_numpy(dtype=np.uint64)
            for index, column in enumerate(source_columns):
                alias = f"__nr_source_{index}__"
                sources[column] = alias
                payload[alias] = child[column].reset_index(drop=True)
        else:
            try:
                child_key = database.primary_keys[relation.child_table]
            except KeyError as exc:
                raise ValueError(
                    f"intermediate table {relation.child_table!r} has no primary key"
                ) from exc
            payload[_NODE] = child[child_key].reset_index(drop=True)
            if payload[_NODE].isna().any():
                raise ValueError(
                    f"primary key {relation.child_table}.{child_key} is missing"
                )
            if payload[_NODE].dropna().duplicated().any():
                raise ValueError(
                    f"primary key {relation.child_table}.{child_key} is not unique"
                )

        joined = payload[payload[_JOIN].notna()].merge(
            mapping[mapping[_NODE].notna()],
            how="inner",
            left_on=_JOIN,
            right_on=_NODE,
            suffixes=("_child", "_parent"),
            sort=False,
            validate="many_to_one",
        )
        child_values = joined[f"{_TIME}_child"].to_numpy(dtype=float)
        parent_values = joined[f"{_TIME}_parent"].to_numpy(dtype=float)
        child_timed = child_time is not None
        joined[_TIME] = _compose_time(
            parent_values,
            child_values,
            accumulated_timed=accumulated_timed,
            step_timed=child_timed,
        )
        accumulated_timed = accumulated_timed or child_timed
        if terminal:
            columns = [_ENTITY, _TIME, _TIE, *sources.values()]
            return joined[columns].reset_index(drop=True), sources
        mapping = joined[[_NODE + "_child", _ENTITY, _TIME]].rename(
            columns={_NODE + "_child": _NODE}
        )
        current = relation.child_table
    raise RuntimeError("relation path must contain at least one edge")


def _temporal_path_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    sources: dict[str, str],
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    valid_time = np.isfinite(events[_TIME])
    if not valid_time.all():
        events = events.loc[valid_time].reset_index(drop=True)
    anchor_keys, event_keys = _aligned_keys(anchors[_ENTITY], events[_ENTITY])
    anchors = anchors.assign(**{_ENTITY: anchor_keys})
    events[_ENTITY] = event_keys
    events = events.loc[events[_ENTITY].isin(anchor_keys.unique())].reset_index(
        drop=True
    )
    if events.empty:
        return _empty_path_values(len(anchors), specs)
    mode_values = _temporal_mode_features(
        anchors,
        events,
        sources,
        [spec for spec in specs if spec.operation == "mode"],
        include_missing_in_modes,
    )
    needed = {spec.column for spec in specs if spec.column is not None}
    states, state_names = _prefix_states(events, sources, needed, specs)
    del events
    upper = _lookup_states(anchors, states, _AT)
    lower: dict[int, pd.DataFrame] = {}
    for window in sorted(
        {spec.window_ns for spec in specs if spec.window_ns is not None}
    ):
        shifted = anchors.assign(**{_AT: anchors[_AT] - float(window)})
        lower[int(window)] = _lookup_states(shifted, states, _AT)

    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.operation == "count":
            values = _state(upper, "rows")
            if spec.window_ns is not None:
                values = values - _state(lower[spec.window_ns], "rows")
            output[spec.name] = values.clip(min=0)
        elif spec.operation == "recency":
            event_time = pd.to_numeric(upper[_TIME], errors="coerce").to_numpy(
                dtype=float
            )
            values = (anchors[_AT].to_numpy(dtype=float) - event_time) / _DAY_NS
            output[spec.name] = values
        elif spec.operation in {"mean_age", "oldest_age"}:
            output[spec.name] = _age_stat(spec.operation, anchors, upper, state_names)
        elif spec.operation == "last":
            output[spec.name] = upper[state_names[(spec.column, "last")]].reset_index(
                drop=True
            )
        elif spec.operation == "mode":
            output[spec.name] = mode_values[spec.name]
        elif spec.operation == "missing_fraction":
            missing = _state(upper, state_names[(spec.column, "missing")])
            rows = _state(upper, "rows")
            output[spec.name] = np.divide(
                missing,
                rows,
                out=np.full(len(rows), np.nan),
                where=rows > 0,
            )
        else:
            output[spec.name] = _numeric_stat(
                spec, upper, lower.get(spec.window_ns), state_names
            )
    return output


def _temporal_mode_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    sources: dict[str, str],
    specs: list[FeatureSpec],
    include_missing: bool,
) -> dict[str, pd.Series]:
    """Sweep strict-past events and resolve mode ties by canonical value."""
    if not specs:
        return {}
    ordered_events = events.sort_values(
        [_ENTITY, _TIME, _TIE], kind="mergesort"
    ).reset_index(drop=True)
    ordered_anchors = anchors.sort_values(
        [_ENTITY, _AT, _ROW], kind="mergesort"
    ).reset_index(drop=True)
    event_groups = ordered_events.groupby(_ENTITY, sort=False).indices
    anchor_groups = ordered_anchors.groupby(_ENTITY, sort=False).indices
    mode_sources: list[np.ndarray] = []
    output_arrays: list[np.ndarray] = []
    for spec in specs:
        if spec.column is None:
            raise RuntimeError("mode requires a source column")
        mode_sources.append(ordered_events[sources[spec.column]].to_numpy(dtype=object))
        output_arrays.append(np.full(len(anchors), None, dtype=object))

    event_times = ordered_events[_TIME].to_numpy(dtype=float)
    anchor_rows = ordered_anchors[_ROW].to_numpy(dtype=np.int64)
    anchor_cutoffs = ordered_anchors[_AT].to_numpy(dtype=float)
    for entity, anchor_positions in anchor_groups.items():
        event_positions = event_groups.get(entity)
        if event_positions is None:
            continue
        counts = [{} for _ in specs]
        values = [{} for _ in specs]
        heaps: list[list[tuple[int, tuple[str, str]]]] = [[] for _ in specs]
        position = 0
        for anchor_position in anchor_positions:
            cutoff = anchor_cutoffs[anchor_position]
            while (
                position < len(event_positions)
                and event_times[event_positions[position]] < cutoff
            ):
                event_position = event_positions[position]
                position += 1
                for index, source in enumerate(mode_sources):
                    category = _category(
                        source[event_position],
                        include_missing=include_missing,
                    )
                    if category is None:
                        continue
                    identity, value = category
                    count = counts[index].get(identity, 0) + 1
                    counts[index][identity] = count
                    values[index][identity] = value
                    heapq.heappush(heaps[index], (-count, identity))
            row = anchor_rows[anchor_position]
            for index, heap in enumerate(heaps):
                while heap and -heap[0][0] != counts[index][heap[0][1]]:
                    heapq.heappop(heap)
                if heap:
                    output_arrays[index][row] = values[index][heap[0][1]]
    return {
        spec.name: pd.Series(result, dtype=object)
        for spec, result in zip(specs, output_arrays, strict=True)
    }


def _category(
    value: object,
    *,
    include_missing: bool = False,
) -> tuple[tuple[str, str], object] | None:
    """Return a deterministic identity and normalized scalar category."""
    if not pd.api.types.is_scalar(value):
        raise ValueError("native categorical values must be scalar")
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        if not include_missing:
            return None
        normalized = _MISSING_CATEGORY
    else:
        normalized = value.item() if isinstance(value, np.generic) else value
        if (
            include_missing
            and isinstance(normalized, str)
            and (
                normalized == _MISSING_CATEGORY
                or normalized.startswith(_CATEGORY_ESCAPE)
            )
        ):
            normalized = f"{_CATEGORY_ESCAPE}{normalized}"
    kind = type(normalized)
    identity = (f"{kind.__module__}.{kind.__qualname__}", repr(normalized))
    return identity, normalized


def _deterministic_mode(
    values: pd.Series,
    *,
    include_missing: bool = False,
) -> object:
    """Return the most frequent scalar with a stable equal-count tie break."""
    counts: dict[tuple[str, str], int] = {}
    originals: dict[tuple[str, str], object] = {}
    for value in values:
        category = _category(value, include_missing=include_missing)
        if category is None:
            continue
        identity, normalized = category
        counts[identity] = counts.get(identity, 0) + 1
        originals[identity] = normalized
    if not counts:
        return None
    winner = min(counts, key=lambda identity: (-counts[identity], identity))
    return originals[winner]


def _prefix_states(
    events: pd.DataFrame,
    sources: dict[str, str],
    needed: set[str | None],
    specs: list[FeatureSpec],
) -> tuple[pd.DataFrame, dict[tuple[str | None, str], str]]:
    events.sort_values([_ENTITY, _TIME, _TIE], kind="mergesort", inplace=True)
    events.reset_index(drop=True, inplace=True)
    ordered = events
    states = ordered[[_ENTITY, _TIME, _TIE]].copy()
    groups = ordered[_ENTITY]
    names: dict[tuple[str | None, str], str] = {
        (None, "rows"): "__nr_state_rows__",
        (None, "time"): _TIME,
    }
    states[names[(None, "rows")]] = (
        ordered.groupby(_ENTITY, sort=False).cumcount() + 1
    ).astype(float)
    path_operations = {spec.operation for spec in specs}
    if "mean_age" in path_operations:
        name = "__nr_state_time_days_sum__"
        names[(None, "time_days_sum")] = name
        states[name] = (
            (ordered[_TIME].astype(float) / _DAY_NS)
            .groupby(groups, sort=False)
            .cumsum()
        )
    if "oldest_age" in path_operations:
        name = "__nr_state_oldest_time__"
        names[(None, "oldest_time")] = name
        states[name] = ordered[_TIME].groupby(groups, sort=False).cummin()
    requested: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        if spec.column is not None:
            requested[spec.column].add(spec.operation)
    for position, column in enumerate(sorted(value for value in needed if value)):
        source = ordered[sources[column]]
        base = f"__nr_state_{position}"
        last_name = f"{base}_last__"
        operations = requested[column]
        if "missing_fraction" in operations:
            name = f"{base}_missing__"
            names[(column, "missing")] = name
            states[name] = (
                source.isna().astype(float).groupby(groups, sort=False).cumsum()
            )
        categorical = any(
            spec.column == column and spec.kind == "categorical" for spec in specs
        )
        if "last" in operations:
            names[(column, "last")] = last_name
            states[last_name] = (
                source.astype(object).groupby(groups, sort=False).ffill()
                if categorical
                else pd.to_numeric(source, errors="coerce")
                .astype(float)
                .groupby(groups, sort=False)
                .ffill()
            )
        if categorical:
            continue

        numeric = pd.to_numeric(source, errors="coerce").astype(float)
        valid = numeric.notna().astype(float)
        numeric_zero = numeric.fillna(0.0)
        additive: list[tuple[str, pd.Series]] = []
        if operations & {"mean", "std", "sum"}:
            additive.extend([("n", valid), ("sum", numeric_zero)])
        if "std" in operations:
            additive.append(("sumsq", numeric_zero * numeric_zero))
        for statistic, values in additive:
            name = f"{base}_{statistic}__"
            names[(column, statistic)] = name
            states[name] = values.groupby(groups, sort=False).cumsum()
        extrema: list[tuple[str, pd.Series]] = []
        if "min" in operations:
            extrema.append(("min", numeric.groupby(groups, sort=False).cummin()))
        if "max" in operations:
            extrema.append(("max", numeric.groupby(groups, sort=False).cummax()))
        for statistic, values in extrema:
            name = f"{base}_{statistic}__"
            names[(column, statistic)] = name
            states[name] = values.groupby(groups, sort=False).ffill()
    del ordered, groups
    states.sort_values([_TIME, _ENTITY, _TIE], kind="mergesort", inplace=True)
    states.reset_index(drop=True, inplace=True)
    return states, names


def _age_stat(
    operation: str,
    anchors: pd.DataFrame,
    upper: pd.DataFrame,
    names: dict[tuple[str | None, str], str],
) -> np.ndarray:
    """Compute event ages from strict-cutoff temporal prefix states."""
    cutoff_days = anchors[_AT].to_numpy(dtype=float) / _DAY_NS
    if operation == "mean_age":
        count = _state(upper, "rows")
        total_days = _state(upper, names[(None, "time_days_sum")])
        mean_days = np.divide(
            total_days,
            count,
            out=np.full(len(total_days), np.nan),
            where=count > 0,
        )
        return cutoff_days - mean_days
    if operation == "oldest_age":
        oldest = pd.to_numeric(
            upper[names[(None, "oldest_time")]], errors="coerce"
        ).to_numpy(dtype=float)
        return (anchors[_AT].to_numpy(dtype=float) - oldest) / _DAY_NS
    raise RuntimeError(f"unsupported age operation {operation!r}")


def _lookup_states(
    anchors: pd.DataFrame, states: pd.DataFrame, at_column: str
) -> pd.DataFrame:
    left = anchors[[_ROW, _ENTITY, at_column]].sort_values(
        [at_column, _ENTITY, _ROW], kind="mergesort"
    )
    merged = pd.merge_asof(
        left,
        states,
        left_on=at_column,
        right_on=_TIME,
        by=_ENTITY,
        allow_exact_matches=False,
        direction="backward",
    )
    return merged.sort_values(_ROW, kind="mergesort").reset_index(drop=True)


def _numeric_stat(
    spec: FeatureSpec,
    upper: pd.DataFrame,
    lower: pd.DataFrame | None,
    names: dict[tuple[str | None, str], str],
) -> np.ndarray:
    if spec.column is None:
        raise RuntimeError(f"{spec.operation} requires a source column")
    if spec.operation in {"min", "max"}:
        return pd.to_numeric(
            upper[names[(spec.column, spec.operation)]], errors="coerce"
        ).to_numpy(dtype=float)
    if spec.operation == "last":
        return pd.to_numeric(
            upper[names[(spec.column, "last")]], errors="coerce"
        ).to_numpy(dtype=float)

    count = _state(upper, names[(spec.column, "n")])
    total = _state(upper, names[(spec.column, "sum")])
    squares = (
        _state(upper, names[(spec.column, "sumsq")])
        if spec.operation == "std"
        else None
    )
    if lower is not None:
        count -= _state(lower, names[(spec.column, "n")])
        total -= _state(lower, names[(spec.column, "sum")])
        if squares is not None:
            squares -= _state(lower, names[(spec.column, "sumsq")])
    if spec.operation == "sum":
        total[count <= 0] = np.nan
        return total
    if spec.operation == "mean":
        return np.divide(
            total,
            count,
            out=np.full(len(total), np.nan),
            where=count > 0,
        )
    if spec.operation == "std":
        if squares is None:
            raise RuntimeError("standard deviation state is missing")
        numerator = squares - np.divide(
            total * total,
            count,
            out=np.zeros(len(total)),
            where=count > 0,
        )
        variance = np.divide(
            np.maximum(numerator, 0.0),
            count - 1,
            out=np.full(len(total), np.nan),
            where=count > 1,
        )
        return np.sqrt(variance)
    raise RuntimeError(f"unsupported native operation {spec.operation!r}")


def _state(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "rows":
        name = "__nr_state_rows__"
    elif name == "time":
        name = _TIME
    if name not in frame:
        raise RuntimeError(f"native prefix state is missing: {name}")
    return (
        pd.to_numeric(frame[name], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        .copy()
    )


def _static_path_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    sources: dict[str, str],
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    anchor_keys, event_keys = _aligned_keys(anchors[_ENTITY], events[_ENTITY])
    anchors = anchors.assign(**{_ENTITY: anchor_keys})
    events = events.assign(**{_ENTITY: event_keys})
    events = events.loc[events[_ENTITY].isin(anchor_keys.unique())].reset_index(
        drop=True
    )
    if events.empty:
        return _empty_path_values(len(anchors), specs)
    ordered = events.sort_values([_ENTITY, _TIE], kind="mergesort")
    result = pd.DataFrame({_ENTITY: anchors[_ENTITY]})
    groups = ordered.groupby(_ENTITY, sort=False)
    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.window_ns is not None or spec.operation in {
            "mean_age",
            "oldest_age",
            "recency",
        }:
            raise RuntimeError("static paths cannot execute temporal operations")
        if spec.operation == "count":
            aggregate = groups.size().rename(spec.name)
        else:
            if spec.column is None:
                raise RuntimeError(f"{spec.operation} requires a source column")
            source = sources[spec.column]
            if spec.operation == "last":
                aggregate = groups[source].last().rename(spec.name)
            elif spec.operation == "mode":
                aggregate = (
                    groups[source]
                    .agg(
                        lambda values: _deterministic_mode(
                            values,
                            include_missing=include_missing_in_modes,
                        )
                    )
                    .rename(spec.name)
                )
            elif spec.operation == "missing_fraction":
                aggregate = (
                    groups[source]
                    .agg(lambda values: values.isna().mean())
                    .rename(spec.name)
                )
            else:
                numeric = ordered.assign(
                    __nr_numeric__=pd.to_numeric(ordered[source], errors="coerce")
                ).groupby(_ENTITY, sort=False)["__nr_numeric__"]
                aggregate = (
                    numeric.sum(min_count=1)
                    if spec.operation == "sum"
                    else getattr(numeric, spec.operation)()
                ).rename(spec.name)
        joined = result.merge(
            aggregate,
            how="left",
            left_on=_ENTITY,
            right_index=True,
            sort=False,
            validate="many_to_one",
        )
        values = joined[spec.name]
        if spec.operation == "count":
            values = values.fillna(0.0).astype(float)
        output[spec.name] = values.reset_index(drop=True)
    return output


def _empty_path_values(
    rows: int, specs: list[FeatureSpec]
) -> dict[str, pd.Series | np.ndarray]:
    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.operation == "count":
            output[spec.name] = np.zeros(rows, dtype=float)
        elif spec.kind == "categorical":
            output[spec.name] = pd.Series([None] * rows, dtype=object)
        else:
            output[spec.name] = np.full(rows, np.nan)
    return output


def _aligned_keys(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Preserve join-key dtypes unless the two sides are incompatible."""
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if left.dtype == right.dtype and not isinstance(left.dtype, pd.CategoricalDtype):
        return left, right
    if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(right):
        if left.isna().any() or right.isna().any():
            return left.astype("Int64"), right.astype("Int64")
        return left.astype(np.int64), right.astype(np.int64)
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return left.astype(float), right.astype(float)
    if pd.api.types.is_string_dtype(left) and pd.api.types.is_string_dtype(right):
        return left.astype(object), right.astype(object)
    return _stable_key_strings(left), _stable_key_strings(right)


def _stable_key_strings(keys: pd.Series) -> pd.Series:
    """String keys that match across a numeric and a text dtype, keeping nulls.

    `astype("string")` writes the whole float 7.0 as ``"7.0"``, which never meets
    the text id ``"7"``, and a left join answers the miss with NaN rather than an
    error. Whole numbers are therefore written as integers. Foreign keys can be
    null, so missing values stay missing instead of failing the conversion.
    """
    if not pd.api.types.is_numeric_dtype(keys):
        return keys.astype("string")
    numeric = keys.astype(float).to_numpy()
    whole = np.isfinite(numeric) & (np.mod(numeric, 1) == 0)
    strings = keys.astype("string")
    strings[whole] = (
        pd.Series(numeric[whole].astype(np.int64)).astype("string").to_numpy()
    )
    return strings


def _time_ns(series: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(
        series.reset_index(drop=True), errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    values = timestamps.astype("int64").to_numpy(dtype=float)
    values[timestamps.isna().to_numpy()] = np.nan
    return values


def _compose_time(
    accumulated: np.ndarray,
    step: np.ndarray,
    *,
    accumulated_timed: bool,
    step_timed: bool,
) -> np.ndarray:
    """Compose declared timestamps while keeping invalid values fail-closed."""
    if accumulated_timed and step_timed:
        # Unlike fmax, maximum propagates NaN. A missing value in any declared
        # time column therefore makes the path row invisible.
        return np.maximum(accumulated, step)
    if accumulated_timed:
        return accumulated.copy()
    if step_timed:
        return step.copy()
    return np.full(len(accumulated), np.nan)


__all__ = ["NativeRelationalFeaturizer"]
