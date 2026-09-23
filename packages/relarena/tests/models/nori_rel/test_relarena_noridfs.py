from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

from relarena.models.nori_rel.feature.dfs.noridfs import NativeFeatureConfig
from relarena.models.nori_rel.native import RelBenchNativeFeaturizer


def _task(**changes: object) -> SimpleNamespace:
    values = {
        "entity_table": "users",
        "entity_col": "user_id",
        "time_col": "cutoff",
        "target_col": "target",
        "timedelta": pd.Timedelta("7 days"),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _database() -> Database:
    users = Table(
        pd.DataFrame(
            {
                "user_id": [1, 2],
                "age": [30.0, 40.0],
                "segment": ["a", "b"],
                "target": [999.0, 999.0],
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col="user_id",
    )
    events = Table(
        pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4],
                "user_id": [1, 1, 1, 2],
                "event_time": pd.to_datetime(
                    ["2026-01-14", "2026-01-15", "2026-01-16", "2026-01-09"]
                ),
                "value": [1.0, 2.0, 3.0, 4.0],
                "target": [14.0, 15.0, 16.0, 9.0],
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users"},
        pkey_col="event_id",
        time_col="event_time",
    )
    return Database({"users": users, "events": events})


def _train_table() -> Table:
    return Table(
        pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "cutoff": pd.to_datetime(["2026-01-15", "2026-01-08", "2026-01-10"]),
                "target": [20.0, 10.0, 30.0],
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users"},
        time_col="cutoff",
    )


def _column(columns: pd.Index, *parts: str) -> str:
    matches = [name for name in columns if all(part in name for part in parts)]
    assert len(matches) == 1, matches
    return matches[0]


def test_adapter_freezes_an_aligned_schema_and_relbench_metadata() -> None:
    task = _task()
    database = _database()
    train = _train_table()
    adapter = RelBenchNativeFeaturizer(
        NativeFeatureConfig(max_paths=8, max_features=64, max_direct_features=8)
    ).fit(task, database, train)

    train_features, categorical = adapter.transform(task, database, train)
    query = Table(
        pd.DataFrame(
            {
                "user_id": [2, 1],
                "cutoff": pd.to_datetime(["2026-01-10", "2026-01-15"]),
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users"},
        time_col="cutoff",
    )
    query_features, query_categorical = adapter.transform(task, database, query)

    assert len(train_features) == 3
    assert len(query_features) == 2
    assert query_features.columns.tolist() == train_features.columns.tolist()
    assert query_categorical == categorical
    assert categorical == [
        "nr|anchor|entity_key",
        "nr|direct|users.segment",
    ]
    assert len(adapter.plan_digest) == 32

    event_count = _column(query_features.columns, "events.user_id", "|rows|count|all")
    # The Jan 15 event is at the cutoff and the Jan 16 event is in the future.
    assert query_features[event_count].tolist() == [1.0, 1.0]


def test_labels_are_excluded_but_same_named_database_history_is_preserved() -> None:
    task = _task()
    database = _database()
    train = _train_table()
    adapter = RelBenchNativeFeaturizer(
        NativeFeatureConfig(max_paths=8, max_features=64, max_direct_features=8)
    ).fit(task, database, train)

    features, _ = adapter.transform(task, database, train)
    assert "nr|direct|users.target" not in features

    history_last = _column(
        features.columns,
        "__nori_rel_target_history__.user_id",
        "|target|last|all",
    )
    assert features[history_last].iloc[0] == 10.0
    assert np.isnan(features[history_last].iloc[1])
    assert np.isnan(features[history_last].iloc[2])

    query = Table(
        pd.DataFrame(
            {
                "user_id": [1, 1],
                "cutoff": pd.to_datetime(["2026-01-15", "2026-01-16"]),
                "target": [-1_000.0, -2_000.0],
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users"},
        time_col="cutoff",
    )
    query_features, _ = adapter.transform(task, database, query)
    assert query_features[history_last].tolist() == [10.0, 20.0]

    database_last = _column(
        features.columns,
        "events.user_id",
        "|target|last|all",
    )
    # Exact-cutoff database rows remain invisible, while prior facts survive.
    assert query_features[database_last].tolist() == [14.0, 15.0]

    without_labels = Table(
        query.df.drop(columns="target"),
        fkey_col_to_pkey_table={"user_id": "users"},
        time_col="cutoff",
    )
    unlabeled_features, _ = adapter.transform(task, database, without_labels)
    pd.testing.assert_frame_equal(query_features, unlabeled_features)


def test_adapter_fails_closed_for_unfitted_or_different_tasks() -> None:
    database = _database()
    train = _train_table()
    adapter = RelBenchNativeFeaturizer()

    with pytest.raises(RuntimeError, match="must be fitted"):
        adapter.transform(_task(), database, train)

    adapter.fit(_task(), database, train)
    with pytest.raises(ValueError, match="does not match"):
        adapter.transform(_task(target_col="another_target"), database, train)


def test_adapter_rejects_ambiguous_target_history() -> None:
    train = _train_table()
    duplicate = Table(
        pd.concat([train.df, train.df.iloc[[0]]], ignore_index=True),
        fkey_col_to_pkey_table=train.fkey_col_to_pkey_table,
        time_col=train.time_col,
    )

    with pytest.raises(ValueError, match="one target per entity and cutoff"):
        RelBenchNativeFeaturizer().fit(_task(), _database(), duplicate)


def test_static_tasks_skip_target_history() -> None:
    task = _task(time_col=None, timedelta=None)
    train = Table(
        pd.DataFrame({"user_id": [1, 2], "target": [1.0, 2.0]}),
        fkey_col_to_pkey_table={"user_id": "users"},
    )
    adapter = RelBenchNativeFeaturizer().fit(task, _database(), train)

    features, _ = adapter.transform(task, _database(), train)

    assert len(features) == 2
    assert not any("target_history" in name for name in features)
