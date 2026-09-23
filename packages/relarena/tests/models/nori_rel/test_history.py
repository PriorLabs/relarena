from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from relarena.models.nori_rel.history import StrictPastHistory


def test_history_is_strict_past_per_entity_and_restores_input_order() -> None:
    rows = pd.DataFrame(
        {
            "driver_id": ["a", "a", "a", "b", "b"],
            "cutoff": pd.to_datetime(
                ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-01"]
            ),
            "target": [30.0, 10.0, 20.0, 200.0, 100.0],
        }
    )
    features = pd.DataFrame({"row": np.arange(len(rows))})
    task = SimpleNamespace(
        entity_col="driver_id",
        time_col="cutoff",
        target_col="target",
    )

    result = StrictPastHistory(2).fit_transform(
        features,
        task,
        SimpleNamespace(df=rows),
    )

    np.testing.assert_array_equal(result["row"], np.arange(len(rows)))
    np.testing.assert_allclose(
        result["target_lag1"],
        [20.0, np.nan, 10.0, 100.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result["target_lag2"],
        [10.0, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result["target_lag1_age_days"],
        [1.0, np.nan, 1.0, 1.0, np.nan],
        equal_nan=True,
    )


def test_transform_uses_only_the_frozen_training_pool() -> None:
    train = pd.DataFrame(
        {
            "driver_id": ["a", "a"],
            "cutoff": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "target": [10.0, 30.0],
        }
    )
    task = SimpleNamespace(
        entity_col="driver_id",
        time_col="cutoff",
        target_col="target",
    )
    history = StrictPastHistory(1)
    history.fit_transform(
        pd.DataFrame({"feature": [0.0, 1.0]}),
        task,
        SimpleNamespace(df=train),
    )
    query = SimpleNamespace(
        df=pd.DataFrame(
            {
                "driver_id": ["a", "a"],
                "cutoff": pd.to_datetime(["2026-01-03", "2026-01-04"]),
                "target": [999.0, 999.0],
            }
        )
    )

    result = history.transform(
        pd.DataFrame({"feature": [2.0, 3.0]}),
        task,
        query,
    )

    # Exact-time train labels and query labels are both excluded.
    np.testing.assert_allclose(result["target_lag1"], [10.0, 30.0])
    np.testing.assert_allclose(result["target_lag1_age_days"], [2.0, 1.0])


def test_history_fails_closed_for_collisions_and_invalid_anchors() -> None:
    train = pd.DataFrame(
        {
            "driver_id": ["a"],
            "cutoff": pd.to_datetime(["2026-01-01"]),
            "target": [10.0],
        }
    )
    task = SimpleNamespace(
        entity_col="driver_id",
        time_col="cutoff",
        target_col="target",
    )
    history = StrictPastHistory(1)
    history.fit_transform(
        pd.DataFrame({"feature": [0.0]}),
        task,
        SimpleNamespace(df=train),
    )

    with pytest.raises(ValueError, match="output columns already exist"):
        history.transform(
            pd.DataFrame({"target_lag1": [0.0]}),
            task,
            SimpleNamespace(df=train),
        )
    with pytest.raises(ValueError, match="query columns are missing"):
        history.transform(
            pd.DataFrame({"feature": [0.0]}),
            task,
            SimpleNamespace(df=train.drop(columns="cutoff")),
        )
    with pytest.raises(ValueError, match="keys and cutoffs must be present"):
        history.transform(
            pd.DataFrame({"feature": [0.0]}),
            task,
            SimpleNamespace(df=train.assign(cutoff=pd.NaT)),
        )
