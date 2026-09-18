"""Opt-in RPI tests using the local TabPFN v3 checkpoint.

Install the local extra and set TABPFN_RUN_LOCAL_TESTS=1 to run these tests.
TabPFN downloads the checkpoint if it is not cached.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from relarena_core.userdb import (
    DatabaseSpec,
    PredictiveContext,
    PredictiveQuery,
    PredictiveQuerySpec,
    PredictiveTaskSpec,
)
from relarena_core.userdb.ingest import TableSource

pytestmark = pytest.mark.skipif(
    os.environ.get("TABPFN_RUN_LOCAL_TESTS") != "1",
    reason="Set TABPFN_RUN_LOCAL_TESTS=1 to run local checkpoint tests.",
)


@pytest.fixture(autouse=True)
def disable_persistent_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "RELARENA_CACHE_DIR",
        "RELARENA_DISABLE_CACHE",
        "RELARENA_DISABLE_FEATURE_CACHE",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize("offset", [0, 1], ids=["at-signup", "after-signup"])
def test_sparse_training_columns_predict_with_local_tabpfn(
    tmp_path: Path, offset: int
) -> None:
    """Temporal censoring produces missing fit values and populated query values."""
    training_dates = pd.date_range(end="2020-10-01", periods=10, freq="30D")
    training_dates -= pd.Timedelta(days=offset)
    customers = pd.DataFrame(
        {
            "customer_id": range(24),
            "created_at": pd.to_datetime(
                list(training_dates.repeat(2)) + ["2020-11-15"] * 2 + ["2021-01-15"] * 2
            ),
            "age": list(range(20, 40)) + [25, 35, 25, 35],
            "user_type": [None] * 20 + ["general", "business"] * 2,
        }
    )
    path = tmp_path / "customers.parquet"
    customers.to_parquet(path, index=False)
    spec = PredictiveQuerySpec(
        database=DatabaseSpec(
            tables={
                "customers": TableSource(
                    path=str(path), pkey="customer_id", time_col="created_at"
                )
            }
        ),
        task=PredictiveTaskSpec(
            entity_table="customers",
            entity_col="customer_id",
            time_col="timestamp",
            target_col="label",
            task_type="binary_classification",
            timedelta="30 days",
            val_timestamp="2020-10-01",
            test_timestamp="2020-12-01",
            query=f"""
                SELECT t.timestamp, c.customer_id,
                       CAST(c.customer_id % 2 AS INTEGER) AS label
                FROM timestamp_df t CROSS JOIN customers c
                WHERE c.created_at + INTERVAL '{offset} days' = t.timestamp
            """,
        ),
    )
    context = PredictiveContext(spec)
    fitted = context.fit("tabpfn-rel-local", n_trials=0)
    predictions = fitted.predict(
        PredictiveQuery(entities=[20, 21], at_timestamp="test_timestamp")
    )
    assert predictions["customer_id"].tolist() == [20, 21]
    assert np.isfinite(predictions["label_pred"]).all()
    assert predictions["label_pred"].between(0, 1).all()


def test_tiny_database_predicts_with_local_tabpfn(
    tmp_path: Path, write_database: Callable[..., Path]
) -> None:
    task = write_database(tmp_path)
    context = PredictiveContext.from_yaml(task, data_dir=tmp_path)
    fitted = context.fit("tabpfn-rel-local", n_trials=0)
    predictions = fitted.predict(
        PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    )
    assert sorted(predictions["customer_id"]) == ["a", "b", "c", "d"]
    assert np.isfinite(predictions["y_pred"]).all()
    assert predictions["y_pred"].between(0, 1).all()
