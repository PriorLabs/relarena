"""Bounded association-to-parent paths and their temporal visibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relarena.models.nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    ForeignKey,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
    RelationPath,
)


def _definition(*, temporal: bool = True) -> AnchorDefinition:
    return AnchorDefinition.create(
        entity_table="facilities",
        entity_column="facility",
        cutoff_column="cutoff" if temporal else None,
        horizon="2 days" if temporal else None,
    )


def _database(
    *,
    association_key: bool = True,
    association_time: bool = True,
) -> DatabaseView:
    facilities_studies = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "facility_id": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            "nct_id": [10, 11, 10, 12, 10, 13, 999, None, 10, 11],
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-04",
                    "invalid",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                ],
                format="mixed",
                errors="coerce",
            ),
        }
    )
    primary_keys = {
        "facilities": "facility_id",
        "studies": "nct_id",
        "sponsors": "sponsor_id",
    }
    if association_key:
        primary_keys["facilities_studies"] = "id"
    time_columns = {"studies": "start_date", "sponsors": "valid_from"}
    if association_time:
        time_columns["facilities_studies"] = "date"
    relationships = [
        ("facilities_studies", "facility_id", "facilities", "facility_id"),
        ("facilities_studies", "nct_id", "studies", "nct_id"),
        ("studies", "sponsor_id", "sponsors", "sponsor_id"),
        ("sponsors", "home_facility_id", "facilities", "facility_id"),
    ]
    return DatabaseView.from_parts(
        tables={
            "facilities": pd.DataFrame(
                {"facility_id": [1, 2], "country": ["US", "CA"]}
            ),
            "facilities_studies": facilities_studies,
            "studies": pd.DataFrame(
                {
                    "nct_id": [10, 11, 12, 13],
                    "sponsor_id": [100, 101, 100, 101],
                    "start_date": pd.to_datetime(
                        [
                            "2019-12-01",
                            "2020-01-03",
                            "2020-01-01",
                            "invalid",
                        ],
                        format="mixed",
                        errors="coerce",
                    ),
                    "enrollment": [100.0, 300.0, 900.0, 700.0],
                    "phase": ["B", "A", "C", "D"],
                    "secret": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "sponsors": pd.DataFrame(
                {
                    "sponsor_id": [100, 101],
                    "home_facility_id": [2, 1],
                    "valid_from": pd.to_datetime(["2019-01-01", "2019-01-01"]),
                    "budget": [1_000.0, 2_000.0],
                }
            ),
        },
        primary_keys=primary_keys,
        foreign_keys=[*relationships, relationships[1]],
        time_columns=time_columns,
    )


def _bridge_feature(
    featurizer: NativeRelationalFeaturizer,
    operation: str,
    *,
    terminal: str = "studies",
    column: str | None = None,
    window: str | None = None,
) -> str:
    window_ns = None if window is None else int(pd.Timedelta(window).value)
    matches = [
        spec.name
        for spec in featurizer.plan.features
        if spec.path is not None
        and spec.path.is_bridge
        and spec.path.terminal_table == terminal
        and spec.operation == operation
        and spec.column == column
        and spec.window_ns == window_ns
    ]
    assert len(matches) == 1, matches
    return matches[0]


def test_bridge_contract_accepts_one_turn_and_rejects_other_shapes() -> None:
    association = ForeignKey("links", "entity_id", "entities", "entity_id")
    parent = ForeignKey("links", "item_id", "items", "item_id")
    grandparent = ForeignKey("items", "group_id", "groups", "group_id")
    path = RelationPath("entities", (association, parent, grandparent), "bridge")

    assert path.is_bridge
    assert path.terminal_table == "groups"
    assert "D:links.entity_id->entities.entity_id" in path.identity
    assert "U:items.group_id->groups.group_id" in path.identity

    with pytest.raises(ValueError, match="descendant and a parent"):
        RelationPath("entities", (association,), "bridge")
    with pytest.raises(ValueError, match="start at the entity"):
        RelationPath("other", (association, parent), "bridge")
    with pytest.raises(ValueError, match="not contiguous"):
        RelationPath(
            "entities",
            (
                association,
                ForeignKey("other", "item_id", "items", "item_id"),
            ),
            "bridge",
        )
    with pytest.raises(ValueError, match="cannot revisit"):
        RelationPath("entities", (association, association), "bridge")
    with pytest.raises(ValueError, match="not contiguous"):
        RelationPath(
            "entities",
            (
                association,
                parent,
                ForeignKey("children", "item_id", "items", "item_id"),
            ),
            "bridge",
        )


def test_bridge_planning_is_bounded_cycle_safe_and_order_independent() -> None:
    database = _database()
    depth_one = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        NativeFeatureConfig(max_depth=1, max_paths=8, max_features=256),
    ).plan
    depth_two = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        NativeFeatureConfig(max_depth=2, max_paths=2, max_features=256),
    ).plan
    config = NativeFeatureConfig(max_depth=3, max_paths=8, max_features=256)
    depth_three = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        config,
    ).plan

    assert not any(path.is_bridge for path in depth_one.paths)
    assert [path.terminal_table for path in depth_two.paths if path.is_bridge] == [
        "studies"
    ]
    assert [path.terminal_table for path in depth_three.paths if path.is_bridge] == [
        "studies",
        "sponsors",
    ]
    assert all(
        len(
            {
                path.entity_table,
                path.edges[0].child_table,
                *(edge.parent_table for edge in path.edges[1:]),
            }
        )
        == len(path.edges) + 1
        for path in depth_three.paths
        if path.is_bridge
    )

    reordered = DatabaseView.from_parts(
        tables={
            name: frame.iloc[::-1].reset_index(drop=True)
            for name, frame in reversed(database.tables.items())
        },
        primary_keys=dict(reversed(database.primary_keys.items())),
        foreign_keys=list(reversed(database.foreign_keys)),
        time_columns=dict(reversed(database.time_columns.items())),
    )
    repeated = NativeRelationalFeaturizer.fit(reordered, _definition(), config).plan
    assert [path.identity for path in repeated.paths] == [
        path.identity for path in depth_three.paths
    ]
    assert repeated.columns == depth_three.columns
    assert repeated.digest == depth_three.digest


def test_bridge_aggregates_occurrences_with_strict_composed_timestamps() -> None:
    database = _database()
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        NativeFeatureConfig(max_depth=2, max_paths=8, max_features=256),
        excluded_columns={"studies": {"secret"}},
    )
    anchors = pd.DataFrame(
        {
            "facility": [1, 2, 99, 1],
            "cutoff": pd.to_datetime(
                ["2020-01-04", "2020-01-04", "2020-01-04", "2020-01-03"]
            ),
        }
    )

    matrix = featurizer.transform(database, anchors)
    count = _bridge_feature(featurizer, "count")
    mean = _bridge_feature(featurizer, "mean", column="enrollment")
    mode = _bridge_feature(featurizer, "mode", column="phase")
    latest = _bridge_feature(featurizer, "last", column="phase")
    window_count = _bridge_feature(featurizer, "count", window="2 days")

    assert matrix.frame[count].tolist() == [3.0, 2.0, 0.0, 2.0]
    np.testing.assert_allclose(
        matrix.frame[mean].to_numpy(),
        [500.0 / 3.0, 200.0, np.nan, 100.0],
        equal_nan=True,
    )
    assert matrix.frame[mode].tolist() == ["B", "A", None, "B"]
    assert matrix.frame[latest].iloc[[0, 1, 3]].tolist() == ["A", "A", "B"]
    assert pd.isna(matrix.frame[latest].iloc[2])
    assert matrix.frame[window_count].tolist() == [2.0, 1.0, 0.0, 2.0]
    assert mode in matrix.categorical_columns
    assert not any("secret" in column for column in matrix.frame)

    changed = _database()
    changed.tables["studies"].loc[:, "secret"] = [999.0, 998.0, 997.0, 996.0]
    reordered = DatabaseView.from_parts(
        tables={
            name: frame.iloc[::-1].reset_index(drop=True)
            for name, frame in reversed(changed.tables.items())
        },
        primary_keys=dict(reversed(changed.primary_keys.items())),
        foreign_keys=list(reversed(changed.foreign_keys)),
        time_columns=dict(reversed(changed.time_columns.items())),
    )
    actual = featurizer.transform(reordered, anchors).frame
    pd.testing.assert_frame_equal(actual, matrix.frame)


def test_bridge_supports_associations_without_primary_keys() -> None:
    database = _database(association_key=False)
    config = NativeFeatureConfig(max_depth=2, max_paths=8, max_features=256)
    first = NativeRelationalFeaturizer.fit(database, _definition(), config)
    anchors = pd.DataFrame(
        {
            "facility": [1, 2],
            "cutoff": pd.to_datetime(["2020-01-04", "2020-01-04"]),
        }
    )
    expected = first.transform(database, anchors).frame
    reordered = DatabaseView.from_parts(
        tables={
            name: frame.iloc[::-1].reset_index(drop=True)
            for name, frame in database.tables.items()
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    second = NativeRelationalFeaturizer.fit(reordered, _definition(), config)
    actual = second.transform(reordered, anchors).frame

    for operation, column in (("count", None), ("mode", "phase"), ("last", "phase")):
        first_name = _bridge_feature(first, operation, column=column)
        second_name = _bridge_feature(second, operation, column=column)
        assert first_name == second_name
        assert actual[second_name].tolist() == expected[first_name].tolist()


def test_temporal_bridge_requires_association_time_but_static_bridge_does_not() -> None:
    database = _database(association_time=False)
    config = NativeFeatureConfig(max_depth=3, max_paths=8, max_features=256)
    temporal = NativeRelationalFeaturizer.fit(database, _definition(), config).plan
    static = NativeRelationalFeaturizer.fit(
        database,
        _definition(temporal=False),
        config,
    ).plan

    assert not any(path.is_bridge for path in temporal.paths)
    assert any(path.is_bridge for path in static.paths)


def test_bridge_rejects_nonunique_parent_keys() -> None:
    database = _database()
    duplicate = DatabaseView.from_parts(
        tables={
            **database.tables,
            "studies": pd.concat(
                [database.tables["studies"], database.tables["studies"].iloc[[0]]],
                ignore_index=True,
            ),
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    featurizer = NativeRelationalFeaturizer.fit(
        duplicate,
        _definition(),
        NativeFeatureConfig(max_depth=2, max_paths=8, max_features=256),
    )
    anchors = pd.DataFrame({"facility": [1], "cutoff": pd.to_datetime(["2020-01-04"])})

    with pytest.raises(ValueError, match=r"studies\.nct_id is not unique"):
        featurizer.transform(duplicate, anchors)
