"""Native aggregation, cutoff visibility, and fitted-schema contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relarena.models.nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    NativeFeatureConfig,
    NativeFeatureMatrix,
    NativeRelationalFeaturizer,
)


def _driver_database() -> DatabaseView:
    return DatabaseView.from_parts(
        tables={
            "drivers": pd.DataFrame(
                {
                    "driver_id": [1, 2],
                    "experience": [20.0, 30.0],
                    "team": ["a", "b"],
                }
            ),
            "results": pd.DataFrame(
                {
                    "result_id": [10, 11, 12, 13],
                    "driver_id": [1, 1, 1, 2],
                    "event_time": pd.to_datetime(
                        ["2020-01-01", "2020-01-03", "2020-01-05", "2020-01-02"],
                        utc=True,
                    ),
                    "score": [1.0, 3.0, 5.0, 10.0],
                    "status": ["first", "second", "third", "only"],
                }
            ),
        },
        primary_keys={"drivers": "driver_id", "results": "result_id"},
        foreign_keys=[("results", "driver_id", "drivers", "driver_id")],
        time_columns={"results": "event_time"},
    )


def _definition() -> AnchorDefinition:
    return AnchorDefinition.create(
        entity_table="drivers",
        entity_column="driver",
        cutoff_column="cutoff",
        horizon="2 days",
    )


def _spec(
    plan: FeaturePlan,
    operation: str,
    *,
    column: str | None = None,
    window: str | None = None,
    terminal: str = "results",
) -> str:
    window_ns = None if window is None else int(pd.Timedelta(window).value)
    matches = [
        feature.name
        for feature in plan.features
        if feature.operation == operation
        and feature.column == column
        and feature.window_ns == window_ns
        and (feature.path is None or feature.path.terminal_table == terminal)
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _fit(database: DatabaseView | None = None) -> NativeRelationalFeaturizer:
    return NativeRelationalFeaturizer.fit(
        database or _driver_database(),
        _definition(),
        NativeFeatureConfig(max_features=100),
    )


def test_manual_point_in_time_aggregates_and_windows() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    anchors = pd.DataFrame(
        {
            "driver": [1, 1, 2],
            "cutoff": pd.to_datetime(
                ["2020-01-03", "2020-01-06", "2020-01-02"], utc=True
            ),
        }
    )

    matrix = featurizer.transform(database, anchors)

    assert isinstance(matrix, NativeFeatureMatrix)
    assert matrix.plan_digest == featurizer.plan.digest
    assert matrix.frame.columns.tolist() == list(featurizer.plan.columns)
    assert matrix.frame["nr|anchor|entity_key"].tolist() == [1, 1, 2]
    assert matrix.frame["nr|anchor|cutoff_year"].tolist() == [2020.0] * 3
    assert matrix.frame["nr|anchor|cutoff_month"].tolist() == [1.0] * 3
    assert matrix.frame["nr|anchor|cutoff_day"].tolist() == [3.0, 6.0, 2.0]
    assert matrix.frame["nr|anchor|cutoff_day_of_week"].tolist() == [4.0, 0.0, 3.0]
    expected_epoch_days = anchors["cutoff"].astype("int64").to_numpy(
        dtype=float
    ) / float(pd.Timedelta(days=1).value)
    np.testing.assert_allclose(
        matrix.frame["nr|anchor|cutoff_epoch_days"], expected_epoch_days
    )
    assert "nr|anchor|entity_key" in matrix.categorical_columns
    assert matrix.frame[_spec(featurizer.plan, "count")].tolist() == [1.0, 3.0, 0.0]
    assert matrix.frame[_spec(featurizer.plan, "last", column="score")].tolist()[
        :2
    ] == [
        1.0,
        5.0,
    ]
    assert matrix.frame[_spec(featurizer.plan, "recency")].tolist()[:2] == [2.0, 1.0]
    assert np.isnan(matrix.frame[_spec(featurizer.plan, "recency")].iloc[2])
    assert matrix.frame[_spec(featurizer.plan, "mean_age")].tolist()[:2] == [
        2.0,
        3.0,
    ]
    assert matrix.frame[_spec(featurizer.plan, "oldest_age")].tolist()[:2] == [
        2.0,
        5.0,
    ]
    assert np.isnan(matrix.frame[_spec(featurizer.plan, "mean_age")].iloc[2])
    assert np.isnan(matrix.frame[_spec(featurizer.plan, "oldest_age")].iloc[2])
    assert matrix.frame[_spec(featurizer.plan, "count", window="2 days")].tolist() == [
        1.0,
        1.0,
        0.0,
    ]
    assert matrix.frame[
        _spec(featurizer.plan, "mean", column="score", window="2 days")
    ].tolist()[:2] == [1.0, 5.0]
    assert matrix.frame[_spec(featurizer.plan, "mean", column="score")].tolist()[
        :2
    ] == [
        1.0,
        3.0,
    ]
    assert matrix.frame[_spec(featurizer.plan, "sum", column="score")].tolist()[:2] == [
        1.0,
        9.0,
    ]
    assert matrix.frame[_spec(featurizer.plan, "std", column="score")].iloc[1] == 2.0
    assert matrix.frame[_spec(featurizer.plan, "min", column="score")].iloc[1] == 1.0
    assert matrix.frame[_spec(featurizer.plan, "max", column="score")].iloc[1] == 5.0


def test_categorical_missing_fraction_is_optional_and_strict_past() -> None:
    original = _driver_database()
    tables = {name: frame.copy() for name, frame in original.tables.items()}
    tables["results"]["status"] = [None, "second", "third", "only"]
    database = DatabaseView.from_parts(
        tables=tables,
        primary_keys=original.primary_keys,
        foreign_keys=original.foreign_keys,
        time_columns=original.time_columns,
    )
    anchors = pd.DataFrame(
        {
            "driver": [1, 1, 2],
            "cutoff": pd.to_datetime(
                ["2020-01-03", "2020-01-06", "2020-01-02"], utc=True
            ),
        }
    )
    default = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        NativeFeatureConfig(max_features=100),
    )
    enabled = NativeRelationalFeaturizer.fit(
        database,
        _definition(),
        NativeFeatureConfig(
            max_features=100,
            include_missing_in_modes=True,
            include_missing_fractions=True,
        ),
    )

    assert not any(
        feature.operation == "missing_fraction" for feature in default.plan.features
    )
    column = _spec(enabled.plan, "missing_fraction", column="status")
    enabled_frame = enabled.transform(database, anchors).frame
    values = enabled_frame[column]
    np.testing.assert_allclose(values, [1.0, 1.0 / 3.0, np.nan], equal_nan=True)
    default_mode = _spec(default.plan, "mode", column="status")
    enabled_mode = _spec(enabled.plan, "mode", column="status")
    assert default.transform(database, anchors).frame[default_mode].iloc[0] is None
    assert enabled_frame[enabled_mode].tolist() == [
        "__nori_rel_missing__",
        "__nori_rel_missing__",
        None,
    ]


def test_static_sum_preserves_missing_numeric_values() -> None:
    database = DatabaseView.from_parts(
        tables={
            "drivers": pd.DataFrame({"driver_id": [1, 2]}),
            "results": pd.DataFrame({"driver_id": [1], "score": [np.nan]}),
        },
        primary_keys={"drivers": "driver_id"},
        foreign_keys=[("results", "driver_id", "drivers", "driver_id")],
    )
    definition = AnchorDefinition.create(
        entity_table="drivers",
        entity_column="driver",
        cutoff_column=None,
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(max_features=100),
    )

    matrix = featurizer.transform(database, pd.DataFrame({"driver": [1, 2]})).frame

    total = _spec(featurizer.plan, "sum", column="score")
    assert matrix[total].isna().all()


def test_static_categorical_missingness_is_optional() -> None:
    database = DatabaseView.from_parts(
        tables={
            "drivers": pd.DataFrame({"driver_id": [1, 2]}),
            "results": pd.DataFrame(
                {
                    "driver_id": [1, 1, 2],
                    "status": [None, "finished", None],
                }
            ),
        },
        primary_keys={"drivers": "driver_id"},
        foreign_keys=[("results", "driver_id", "drivers", "driver_id")],
    )
    definition = AnchorDefinition.create(
        entity_table="drivers",
        entity_column="driver",
        cutoff_column=None,
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(
            max_features=100,
            include_missing_in_modes=True,
            include_missing_fractions=True,
        ),
    )

    matrix = featurizer.transform(database, pd.DataFrame({"driver": [1, 2]})).frame

    fraction = _spec(featurizer.plan, "missing_fraction", column="status")
    mode = _spec(featurizer.plan, "mode", column="status")
    assert matrix[fraction].tolist() == [0.5, 1.0]
    assert matrix[mode].tolist() == ["__nori_rel_missing__"] * 2


def test_missing_category_does_not_collide_with_database_values() -> None:
    database = DatabaseView.from_parts(
        tables={
            "drivers": pd.DataFrame({"driver_id": [1]}),
            "results": pd.DataFrame(
                {
                    "driver_id": [1, 1, 1, 1],
                    "status": [None, "__nori_rel_missing__", "z", "z"],
                }
            ),
        },
        primary_keys={"drivers": "driver_id"},
        foreign_keys=[("results", "driver_id", "drivers", "driver_id")],
    )
    definition = AnchorDefinition.create(
        entity_table="drivers",
        entity_column="driver",
        cutoff_column=None,
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(
            max_features=100,
            include_missing_in_modes=True,
        ),
    )

    matrix = featurizer.transform(database, pd.DataFrame({"driver": [1]})).frame

    mode = _spec(featurizer.plan, "mode", column="status")
    assert matrix[mode].tolist() == ["z"]


def test_same_time_and_future_rows_never_change_earlier_features() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-03"], utc=True),
        }
    )
    before = featurizer.transform(database, anchors).frame

    changed_results = pd.concat(
        [
            database.tables["results"],
            pd.DataFrame(
                {
                    "result_id": [20, 21],
                    "driver_id": [1, 1],
                    "event_time": pd.to_datetime(
                        ["2020-01-03", "2021-01-01"], utc=True
                    ),
                    "score": [999.0, 9999.0],
                    "status": ["same", "future"],
                }
            ),
        ],
        ignore_index=True,
    )
    changed = DatabaseView.from_parts(
        tables={**database.tables, "results": changed_results},
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    after = featurizer.transform(changed, anchors).frame

    pd.testing.assert_frame_equal(after, before)


def test_invalid_declared_child_time_is_never_visible() -> None:
    database = DatabaseView.from_parts(
        tables={
            "drivers": pd.DataFrame(
                {
                    "driver_id": [1],
                    "created_at": pd.to_datetime(["2020-01-01"], utc=True),
                }
            ),
            "results": pd.DataFrame(
                {
                    "driver_id": [1],
                    "event_time": [pd.NaT],
                    "score": [99.0],
                }
            ),
        },
        primary_keys={"drivers": "driver_id"},
        foreign_keys=[("results", "driver_id", "drivers", "driver_id")],
        time_columns={"drivers": "created_at", "results": "event_time"},
    )
    featurizer = _fit(database)
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-02"], utc=True),
        }
    )

    matrix = featurizer.transform(database, anchors).frame

    assert matrix[_spec(featurizer.plan, "count")].iloc[0] == 0.0
    assert np.isnan(matrix[_spec(featurizer.plan, "mean", column="score")].iloc[0])


def test_windows_are_relative_to_each_anchor_cutoff() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-02-01"], utc=True),
        }
    )

    matrix = featurizer.transform(database, anchors).frame

    two_day_count = _spec(featurizer.plan, "count", window="2 days")
    two_day_mean = _spec(featurizer.plan, "mean", column="score", window="2 days")
    assert matrix[two_day_count].iloc[0] == 0.0
    assert np.isnan(matrix[two_day_mean].iloc[0])
    # Recency remains relative to the requested prediction cutoff.
    assert matrix[_spec(featurizer.plan, "recency")].iloc[0] == 27.0


def test_unrelated_future_rows_do_not_move_an_entity_window() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-10"], utc=True),
        }
    )
    before = featurizer.transform(database, anchors).frame
    future = pd.DataFrame(
        {
            "result_id": [99],
            "driver_id": [2],
            "event_time": pd.to_datetime(["2020-01-20"], utc=True),
            "score": [999.0],
            "status": ["future"],
        }
    )
    changed = DatabaseView.from_parts(
        tables={
            **database.tables,
            "results": pd.concat(
                [database.tables["results"], future], ignore_index=True
            ),
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )

    after = featurizer.transform(changed, anchors).frame

    pd.testing.assert_frame_equal(after, before)


def test_timestamp_storage_unit_does_not_change_cutoff_semantics() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    nanosecond = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-06"], utc=True),
        }
    )
    second = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.Series(pd.Timestamp("2020-01-06", tz="UTC"), index=[0]),
        }
    )

    expected = featurizer.transform(database, nanosecond).frame
    actual = featurizer.transform(database, second).frame

    pd.testing.assert_frame_equal(actual, expected)


def test_nullable_and_numpy_integer_keys_are_aligned_without_string_casts() -> None:
    database = _driver_database()
    nullable_results = database.tables["results"].astype({"driver_id": "Int64"})
    nullable = DatabaseView.from_parts(
        tables={**database.tables, "results": nullable_results},
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    anchors = pd.DataFrame(
        {
            "driver": pd.Series([1, 2], dtype="int64"),
            "cutoff": pd.to_datetime(["2020-01-06", "2020-01-03"], utc=True),
        }
    )
    featurizer = _fit(nullable)

    matrix = featurizer.transform(nullable, anchors).frame

    assert matrix[_spec(featurizer.plan, "count")].tolist() == [3.0, 1.0]


def test_frame_and_anchor_order_do_not_change_features() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    anchors = pd.DataFrame(
        {
            "driver": [1, 2, 1],
            "cutoff": pd.to_datetime(
                ["2020-01-06", "2020-01-03", "2020-01-03"], utc=True
            ),
        }
    )
    expected = featurizer.transform(database, anchors).frame
    reordered = DatabaseView.from_parts(
        tables={
            name: frame.sample(frac=1, random_state=4)
            .reset_index(drop=True)
            .loc[:, list(reversed(frame.columns))]
            for name, frame in reversed(database.tables.items())
        },
        primary_keys=dict(reversed(database.primary_keys.items())),
        foreign_keys=list(reversed(database.foreign_keys)),
        time_columns=dict(reversed(database.time_columns.items())),
    )

    actual = featurizer.transform(reordered, anchors).frame
    pd.testing.assert_frame_equal(actual, expected)

    order = [2, 0, 1]
    shuffled = anchors.iloc[order].reset_index(drop=True)
    shuffled_features = featurizer.transform(reordered, shuffled).frame
    pd.testing.assert_frame_equal(
        shuffled_features,
        expected.iloc[order].reset_index(drop=True),
    )
    assert shuffled_features["nr|anchor|entity_key"].tolist() == [1, 1, 2]


def test_multi_hop_paths_do_not_multiply_sibling_rows() -> None:
    database = DatabaseView.from_parts(
        tables={
            "customers": pd.DataFrame({"customer_id": [1], "segment": ["a"]}),
            "orders": pd.DataFrame(
                {
                    "order_id": [10, 11],
                    "customer_id": [1, 1],
                    "ordered_at": pd.to_datetime(
                        ["2020-01-01", "2020-01-05"], utc=True
                    ),
                    "total": [20.0, 30.0],
                }
            ),
            "items": pd.DataFrame(
                {
                    "item_id": [100, 101, 102, 103],
                    "order_id": [10, 10, 11, 11],
                    "created_at": pd.to_datetime(
                        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-10"],
                        utc=True,
                    ),
                    "value": [1.0, 2.0, 3.0, 1000.0],
                }
            ),
            "notes": pd.DataFrame(
                {
                    "note_id": [200, 201, 202],
                    "order_id": [10, 10, 11],
                    "created_at": pd.to_datetime(
                        ["2020-01-02", "2020-01-04", "2020-01-07"], utc=True
                    ),
                    "length": [4.0, 5.0, 6.0],
                }
            ),
        },
        primary_keys={
            "customers": "customer_id",
            "orders": "order_id",
            "items": "item_id",
            "notes": "note_id",
        },
        foreign_keys=[
            ("orders", "customer_id", "customers", "customer_id"),
            ("items", "order_id", "orders", "order_id"),
            ("notes", "order_id", "orders", "order_id"),
        ],
        time_columns={
            "orders": "ordered_at",
            "items": "created_at",
            "notes": "created_at",
        },
    )
    definition = AnchorDefinition.create(
        entity_table="customers",
        entity_column="customer",
        cutoff_column="cutoff",
        horizon="2 days",
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        definition,
        NativeFeatureConfig(max_depth=2, max_features=256),
    )
    anchors = pd.DataFrame(
        {
            "customer": [1],
            "cutoff": pd.to_datetime(["2020-01-10"], utc=True),
        }
    )

    matrix = featurizer.transform(database, anchors).frame

    item_count = _spec(featurizer.plan, "count", terminal="items")
    item_sum = _spec(featurizer.plan, "sum", column="value", terminal="items")
    order_count = _spec(featurizer.plan, "count", terminal="orders")
    assert matrix[item_count].iloc[0] == 3.0
    assert matrix[item_sum].iloc[0] == 6.0
    assert matrix[order_count].iloc[0] == 2.0


def test_transform_fails_closed_on_schema_or_anchor_drift() -> None:
    database = _driver_database()
    featurizer = _fit(database)
    changed = DatabaseView.from_parts(
        tables={
            **database.tables,
            "results": database.tables["results"].drop(columns="score"),
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-03"], utc=True),
        }
    )

    with pytest.raises(ValueError, match="schema does not match"):
        featurizer.transform(changed, anchors)
    with pytest.raises(ValueError, match="anchor columns are missing"):
        featurizer.transform(database, anchors.drop(columns="cutoff"))

    changed_dtype = DatabaseView.from_parts(
        tables={
            **database.tables,
            "results": database.tables["results"].astype({"score": "string"}),
        },
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    with pytest.raises(ValueError, match="schema does not match"):
        featurizer.transform(changed_dtype, anchors)


def test_last_without_terminal_primary_key_is_row_order_independent() -> None:
    database = _driver_database()
    results = database.tables["results"].drop(columns="result_id")
    duplicate_time = pd.concat(
        [
            results,
            pd.DataFrame(
                {
                    "driver_id": [1],
                    "event_time": pd.to_datetime(["2020-01-05"], utc=True),
                    "score": [7.0],
                    "status": ["same-time"],
                }
            ),
        ],
        ignore_index=True,
    )
    reordered = DatabaseView.from_parts(
        tables={
            **database.tables,
            "results": duplicate_time.iloc[::-1].reset_index(drop=True),
        },
        primary_keys={"drivers": "driver_id"},
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    original = DatabaseView.from_parts(
        tables={**database.tables, "results": duplicate_time},
        primary_keys={"drivers": "driver_id"},
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-06"], utc=True),
        }
    )
    featurizer = _fit(original)

    expected = featurizer.transform(original, anchors).frame
    actual = featurizer.transform(reordered, anchors).frame

    pd.testing.assert_frame_equal(actual, expected)


def test_no_primary_key_tie_ignores_excluded_and_nested_columns() -> None:
    database = _driver_database()
    results = database.tables["results"].drop(columns="result_id").copy()
    results["excluded_target"] = [1, 2, 3, 4]
    results["nested_payload"] = [{"row": value} for value in range(len(results))]
    original = DatabaseView.from_parts(
        tables={**database.tables, "results": results},
        primary_keys={"drivers": "driver_id"},
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    changed_results = results.copy()
    changed_results["excluded_target"] = [40, 30, 20, 10]
    changed_results["nested_payload"] = [
        {"changed": value} for value in range(len(results))
    ]
    changed = DatabaseView.from_parts(
        tables={**database.tables, "results": changed_results},
        primary_keys={"drivers": "driver_id"},
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )
    featurizer = NativeRelationalFeaturizer.fit(
        original,
        _definition(),
        NativeFeatureConfig(max_features=100),
        excluded_columns={"results": {"excluded_target"}},
    )
    anchors = pd.DataFrame(
        {
            "driver": [1],
            "cutoff": pd.to_datetime(["2020-01-06"], utc=True),
        }
    )

    expected = featurizer.transform(original, anchors).frame
    actual = featurizer.transform(changed, anchors).frame

    assert not any("nested_payload" in column for column in featurizer.plan.columns)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "numeric",
    [
        pd.Series([7.0, 8.0, np.nan]),
        pd.Series([7, 8, pd.NA], dtype="Int64"),
    ],
)
def test_numeric_keys_meet_text_keys_across_the_dtype_gap(numeric: pd.Series) -> None:
    """A whole-number id joins its text twin; a null foreign key stays null."""
    from relarena.models.nori_rel.feature.dfs.noridfs.engine import _aligned_keys

    left, right = _aligned_keys(numeric, pd.Series(["7", "8", "9"], dtype=object))

    assert left.tolist()[:2] == ["7", "8"]
    assert pd.isna(left.iloc[2])
    assert right.tolist() == ["7", "8", "9"]
    assert set(left.dropna()) <= set(right)


def test_integer_keys_past_float_precision_stay_exact() -> None:
    """An id past 2**53 must not round through float and collide."""
    from relarena.models.nori_rel.feature.dfs.noridfs.engine import _aligned_keys

    big = 2**53 + 1
    left, right = _aligned_keys(
        pd.Series([big, big + 1], dtype="int64"),
        pd.Series([str(big), str(big + 1)], dtype=object),
    )

    assert left.tolist() == [str(big), str(big + 1)]
    assert left.tolist() == right.tolist()
