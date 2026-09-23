"""Deterministic native feature plans and search budgets."""

from __future__ import annotations

import pandas as pd
import pytest

from relarena.models.nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
)
from relarena.models.nori_rel.feature.dfs.noridfs import planner as planner_module


class _StopAfterLimit:
    def __iter__(self):
        yield "a"
        yield "b"
        yield "c"
        raise AssertionError("cardinality scan did not stop at limit + 1")


def _database(*, values: tuple[float, ...] = (1.0, 2.0)) -> DatabaseView:
    return DatabaseView.from_parts(
        tables={
            "accounts": pd.DataFrame(
                {
                    "account_id": [1, 2],
                    "region": ["west", "east"],
                    "target": [10.0, 20.0],
                }
            ),
            "events": pd.DataFrame(
                {
                    "event_id": list(range(10, 10 + len(values))),
                    "account_id": [1] * len(values),
                    "timestamp": pd.date_range(
                        "2020-01-01", periods=len(values), tz="UTC"
                    ),
                    "value": values,
                }
            ),
            "timeless": pd.DataFrame(
                {
                    "row_id": [1, 2],
                    "account_id": [1, 2],
                    "static_value": [4.0, 5.0],
                }
            ),
        },
        primary_keys={
            "accounts": "account_id",
            "events": "event_id",
            "timeless": "row_id",
        },
        foreign_keys=[
            ("timeless", "account_id", "accounts", "account_id"),
            ("events", "account_id", "accounts", "account_id"),
        ],
        time_columns={"events": "timestamp"},
    )


def _anchors(cutoff: bool = True) -> AnchorDefinition:
    return AnchorDefinition.create(
        entity_table="accounts",
        entity_column="account",
        cutoff_column="cutoff" if cutoff else None,
        horizon="7 days" if cutoff else None,
    )


def test_plan_is_fit_frozen_bounded_and_value_independent() -> None:
    config = NativeFeatureConfig(
        max_depth=2,
        max_paths=3,
        max_features=24,
        max_direct_features=1,
        window_multipliers=(1, 4),
    )
    first = NativeRelationalFeaturizer.fit(
        _database(),
        _anchors(),
        config,
        excluded_columns={"accounts": {"target"}},
    ).plan
    second = NativeRelationalFeaturizer.fit(
        _database(values=(100.0, 200.0, 300.0)),
        _anchors(),
        config,
        excluded_columns={"accounts": {"target"}},
    ).plan

    assert len(first.features) == config.max_features
    assert first.columns[:6] == (
        "nr|anchor|entity_key",
        "nr|anchor|cutoff_epoch_days",
        "nr|anchor|cutoff_year",
        "nr|anchor|cutoff_month",
        "nr|anchor|cutoff_day",
        "nr|anchor|cutoff_day_of_week",
    )
    assert first.columns == second.columns
    assert first.digest == second.digest
    assert all("target" not in feature.name for feature in first.features)
    assert [path.terminal_table for path in first.paths] == ["events"]
    assert {feature.window_ns for feature in first.features} >= {
        int(pd.Timedelta("7 days").value)
    }


def test_timeless_collections_are_only_used_without_temporal_cutoffs() -> None:
    temporal = NativeRelationalFeaturizer.fit(_database(), _anchors()).plan
    static = NativeRelationalFeaturizer.fit(_database(), _anchors(False)).plan

    assert "timeless" not in {path.terminal_table for path in temporal.paths}
    assert "timeless" in {path.terminal_table for path in static.paths}


def test_direct_only_database_respects_one_feature_budget() -> None:
    database = DatabaseView.from_parts(
        tables={"entity": pd.DataFrame({"id": [1], "value": [2.0]})},
        primary_keys={"entity": "id"},
    )
    definition = AnchorDefinition.create(
        entity_table="entity",
        entity_column="entity_id",
        cutoff_column=None,
    )
    plan = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(max_features=1),
    ).plan

    assert len(plan.features) == 1
    assert plan.features[0].name == "nr|anchor|entity_key"
    assert plan.categorical_columns == ("nr|anchor|entity_key",)


def test_direct_budget_prioritizes_numeric_and_datetime_columns() -> None:
    database = DatabaseView.from_parts(
        tables={
            "entity": pd.DataFrame(
                {
                    "id": [1],
                    "a_category": ["low"],
                    "y_datetime": pd.to_datetime(["2020-01-01"], utc=True),
                    "z_numeric": [2.0],
                }
            )
        },
        primary_keys={"entity": "id"},
    )
    definition = AnchorDefinition.create(
        entity_table="entity",
        entity_column="entity_id",
        cutoff_column=None,
    )

    plan = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(max_features=3, max_direct_features=2),
    ).plan

    assert plan.columns == (
        "nr|anchor|entity_key",
        "nr|direct|entity.y_datetime",
        "nr|direct|entity.z_numeric",
    )


def test_direct_category_budget_keeps_small_domains_before_text() -> None:
    database = DatabaseView.from_parts(
        tables={
            "entity": pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "text": ["a", "b", "c", "d"],
                    "segment": ["x", "x", "y", "y"],
                }
            )
        },
        primary_keys={"entity": "id"},
    )
    definition = AnchorDefinition.create(
        entity_table="entity",
        entity_column="entity_id",
        cutoff_column=None,
    )

    plan = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(
            max_features=3,
            max_direct_features=2,
            max_direct_categorical_cardinality=2,
        ),
    ).plan

    assert plan.columns == (
        "nr|anchor|entity_key",
        "nr|direct|entity.segment",
    )


def test_cardinality_scan_stops_after_exceeding_limit() -> None:
    assert planner_module._bounded_cardinality(_StopAfterLimit(), 2) is None


def test_mixed_nested_object_column_is_rejected_independent_of_row_order() -> None:
    rows = pd.DataFrame({"id": [1, 2], "mixed": ["scalar", ["nested"]]})
    definition = AnchorDefinition.create(
        entity_table="entity",
        entity_column="entity_id",
        cutoff_column=None,
    )

    plans = [
        NativeRelationalFeaturizer.fit(
            DatabaseView.from_parts(
                tables={"entity": frame.reset_index(drop=True)},
                primary_keys={"entity": "id"},
            ),
            definition,
        ).plan
        for frame in (rows, rows.iloc[::-1])
    ]

    assert plans[0].columns == ("nr|anchor|entity_key",)
    assert plans[0].columns == plans[1].columns
    assert plans[0].digest == plans[1].digest


def test_categorical_path_aggregates_have_a_fitted_cardinality_cap() -> None:
    database = _database(values=(1.0, 2.0, 3.0, 4.0, 5.0))
    events = database.tables["events"].assign(
        category=["a", "b", "a", "b", "a"],
        free_text=[f"description {index}" for index in range(5)],
    )
    bounded = DatabaseView.from_parts(
        tables={**database.tables, "events": events},
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )

    plan = NativeRelationalFeaturizer.fit(
        bounded,
        _anchors(),
        NativeFeatureConfig(
            max_features=128,
            max_categorical_cardinality=2,
        ),
    ).plan

    categorical = {
        (feature.column, feature.operation)
        for feature in plan.features
        if feature.kind == "categorical" and feature.path is not None
    }
    assert ("category", "mode") in categorical
    assert ("category", "last") in categorical
    assert not any(column == "free_text" for column, _ in categorical)


def test_snapshot_shaped_parts_reject_composite_primary_keys() -> None:
    with pytest.raises(ValueError, match="one primary-key column"):
        DatabaseView.from_parts(
            tables={"entity": pd.DataFrame({"left": [1], "right": [2]})},
            primary_keys={"entity": ["left", "right"]},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"max_depth": 0},
        {"max_paths": 0},
        {"max_features": 0},
        {"max_direct_features": -1},
        {"max_direct_categorical_cardinality": 0},
        {"max_categorical_cardinality": 0},
        {"window_multipliers": (1, 1)},
    ],
)
def test_invalid_search_budgets_fail_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        NativeFeatureConfig(**changes)
