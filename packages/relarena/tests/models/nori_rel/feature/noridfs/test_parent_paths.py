"""Cardinality-preserving, point-in-time parent lookups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relarena.models.nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
)


def _database() -> DatabaseView:
    return DatabaseView.from_parts(
        tables={
            "orders": pd.DataFrame(
                {
                    "order_id": [100, 101],
                    "customer_id": [10, 20],
                    "created_at": pd.to_datetime(
                        ["2020-01-01", "2020-01-01"], utc=True
                    ),
                    "amount": [12.0, 20.0],
                }
            ),
            "customers": pd.DataFrame(
                {
                    "customer_id": [10, 20],
                    "country_id": [1, 2],
                    "valid_from": pd.to_datetime(
                        ["2020-01-02", "2020-01-04"], utc=True
                    ),
                    "lifetime_value": [100.0, 200.0],
                    "tier": ["gold", "silver"],
                }
            ),
            "countries": pd.DataFrame(
                {
                    "country_id": [1, 2],
                    "updated_at": pd.to_datetime(
                        ["2020-01-01", "2020-01-20"], utc=True
                    ),
                    "tax_rate": [0.1, 0.2],
                    "region": ["north", "south"],
                }
            ),
            "items": pd.DataFrame(
                {
                    "item_id": [1, 2],
                    "order_id": [100, 101],
                    "event_time": pd.to_datetime(
                        ["2020-01-02", "2020-01-03"], utc=True
                    ),
                    "quantity": [2.0, 4.0],
                }
            ),
        },
        primary_keys={
            "orders": "order_id",
            "customers": "customer_id",
            "countries": "country_id",
            "items": "item_id",
        },
        foreign_keys=[
            ("orders", "customer_id", "customers", "customer_id"),
            ("customers", "country_id", "countries", "country_id"),
            ("items", "order_id", "orders", "order_id"),
        ],
        time_columns={
            "orders": "created_at",
            "customers": "valid_from",
            "countries": "updated_at",
            "items": "event_time",
        },
    )


def _definition() -> AnchorDefinition:
    return AnchorDefinition.create(
        entity_table="orders",
        entity_column="order",
        cutoff_column="cutoff",
        horizon="2 days",
    )


def _feature(
    featurizer: NativeRelationalFeaturizer,
    operation: str,
    *,
    terminal: str,
    column: str | None = None,
) -> str:
    matches = [
        spec.name
        for spec in featurizer.plan.features
        if spec.operation == operation
        and spec.column == column
        and spec.window_ns is None
        and spec.path is not None
        and spec.path.terminal_table == terminal
    ]
    assert len(matches) == 1, matches
    return matches[0]


def test_parent_lookups_are_multihop_cardinality_preserving_and_point_in_time() -> None:
    database = _database()
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        NativeFeatureConfig(max_depth=2, max_paths=8, max_features=256),
    )
    anchors = pd.DataFrame(
        {
            "order": [100, 101, 101, 101, 999],
            "cutoff": pd.to_datetime(
                [
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-20",
                    "2020-01-21",
                    "2020-01-21",
                ],
                utc=True,
            ),
        }
    )

    matrix = featurizer.transform(database, anchors)

    customer_tier = _feature(featurizer, "lookup", terminal="customers", column="tier")
    customer_value = _feature(
        featurizer, "lookup", terminal="customers", column="lifetime_value"
    )
    country_region = _feature(
        featurizer, "lookup", terminal="countries", column="region"
    )
    country_tax = _feature(
        featurizer, "lookup", terminal="countries", column="tax_rate"
    )
    item_count = _feature(featurizer, "count", terminal="items")

    assert len(matrix.frame) == len(anchors)
    assert matrix.frame[customer_tier].tolist() == [
        "gold",
        None,
        "silver",
        "silver",
        None,
    ]
    assert matrix.frame[country_region].tolist() == ["north", None, None, "south", None]
    np.testing.assert_allclose(
        matrix.frame[customer_value].to_numpy(),
        [100.0, np.nan, 200.0, 200.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        matrix.frame[country_tax].to_numpy(),
        [0.1, np.nan, np.nan, 0.2, np.nan],
        equal_nan=True,
    )
    assert matrix.frame[item_count].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert customer_tier in matrix.categorical_columns
    assert country_region in matrix.categorical_columns


def test_parent_paths_are_order_independent_and_schema_frozen() -> None:
    database = _database()
    featurizer = NativeRelationalFeaturizer.fit(database, _definition())
    anchors = pd.DataFrame(
        {
            "order": [101, 100],
            "cutoff": pd.to_datetime(["2020-01-21", "2020-01-03"], utc=True),
        }
    )
    expected = featurizer.transform(database, anchors).frame
    reordered = DatabaseView.from_parts(
        tables={
            name: frame.iloc[::-1].reset_index(drop=True)
            for name, frame in reversed(database.tables.items())
        },
        primary_keys=dict(reversed(database.primary_keys.items())),
        foreign_keys=list(reversed(database.foreign_keys)),
        time_columns=dict(reversed(database.time_columns.items())),
    )

    actual = featurizer.transform(reordered, anchors).frame

    pd.testing.assert_frame_equal(actual, expected)
    changed = DatabaseView.from_parts(
        tables={
            **database.tables,
            "countries": database.tables["countries"].drop(columns="region"),
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    with pytest.raises(ValueError, match="schema does not match"):
        featurizer.transform(changed, anchors)


def test_parent_lookup_rejects_a_nonunique_referred_key() -> None:
    database = _database()
    duplicated = DatabaseView.from_parts(
        tables={
            **database.tables,
            "customers": pd.concat(
                [
                    database.tables["customers"],
                    database.tables["customers"].iloc[[0]],
                ],
                ignore_index=True,
            ),
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    featurizer = NativeRelationalFeaturizer.fit(duplicated, _definition())
    anchors = pd.DataFrame(
        {
            "order": [100],
            "cutoff": pd.to_datetime(["2020-01-03"], utc=True),
        }
    )

    with pytest.raises(ValueError, match=r"customers\.customer_id is not unique"):
        featurizer.transform(duplicated, anchors)


def test_invalid_declared_parent_time_hides_the_lookup() -> None:
    database = _database()
    customers = database.tables["customers"].copy()
    customers.loc[0, "valid_from"] = pd.NaT
    changed = DatabaseView.from_parts(
        tables={**database.tables, "customers": customers},
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    featurizer = NativeRelationalFeaturizer.fit(changed, _definition())
    anchors = pd.DataFrame(
        {
            "order": [100],
            "cutoff": pd.to_datetime(["2020-01-03"], utc=True),
        }
    )

    matrix = featurizer.transform(changed, anchors).frame
    customer_tier = _feature(featurizer, "lookup", terminal="customers", column="tier")
    customer_value = _feature(
        featurizer, "lookup", terminal="customers", column="lifetime_value"
    )

    assert matrix[customer_tier].iloc[0] is None
    assert np.isnan(matrix[customer_value].iloc[0])


def test_cycles_stop_at_visited_tables_and_path_budget_is_deterministic() -> None:
    tables = {
        "accounts": pd.DataFrame({"account_id": [1], "group_id": [10], "value": [2.0]}),
        "groups": pd.DataFrame(
            {"group_id": [10], "owner_id": [1], "label": ["primary"]}
        ),
    }
    relations = [
        ("accounts", "group_id", "groups", "group_id"),
        ("groups", "owner_id", "accounts", "account_id"),
    ]
    database = DatabaseView.from_parts(
        tables=tables,
        primary_keys={"accounts": "account_id", "groups": "group_id"},
        foreign_keys=relations,
    )
    definition = AnchorDefinition.create(
        entity_table="accounts",
        entity_column="account",
        cutoff_column=None,
    )
    config = NativeFeatureConfig(
        max_depth=8,
        max_paths=8,
        max_features=16,
        max_direct_features=0,
    )

    first = NativeRelationalFeaturizer.fit(database, definition, config).plan
    reordered = DatabaseView.from_parts(
        tables=dict(reversed(tables.items())),
        primary_keys={"groups": "group_id", "accounts": "account_id"},
        foreign_keys=list(reversed(relations)),
    )
    second = NativeRelationalFeaturizer.fit(reordered, definition, config).plan

    assert len(first.paths) == 2
    assert {path.direction for path in first.paths} == {"parent", "descendant"}
    assert all(len(path.edges) == 1 for path in first.paths)
    assert [path.identity for path in first.paths] == [
        path.identity for path in second.paths
    ]
    assert first.columns == second.columns
    assert first.digest == second.digest

    bounded = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(
            max_depth=8,
            max_paths=1,
            max_features=2,
            max_direct_features=0,
        ),
    ).plan
    assert len(bounded.paths) == 1
    assert bounded.paths[0].direction == "parent"
    assert len(bounded.features) == 2
    assert bounded.features[-1].operation == "lookup"
