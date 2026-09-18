"""Exercise all-missing training features through a one-table RPI database.

Training anchors coincide with customer creation. DFS excludes rows at the cutoff,
so customer features are missing at fit. At prediction, older customer records
supply strings to those columns. The fitted feature selection must exclude the
all-missing training columns so prediction succeeds.

Pass --control to create training customers one day before their training anchors.
The source column is still missing for those customers, but DFS stringifies its
nulls; it reaches TabPFN as categorical and prediction succeeds. Labels are
synthetic, based on customer ID parity; this example does not measure accuracy.

Run from the repository root after installing the tabpfn-rel-local extra:
    OMP_NUM_THREADS=1 uv run --no-sync python examples/rpi_sparse_column.py

Uses the local v3 checkpoint, downloading it if it is not already cached.
"""

from __future__ import annotations

import argparse
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from relarena_core.userdb import (
    DatabaseSpec,
    PredictiveContext,
    PredictiveQuery,
    PredictiveQuerySpec,
    PredictiveTaskSpec,
)
from relarena_core.userdb.ingest import TableSource


def main() -> None:
    """Fit on customers with missing categories and predict for new customers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control",
        action="store_true",
        help="Create training customers one day before their training anchors.",
    )
    args = parser.parse_args()
    offset = int(args.control)
    print(f"TabPFN {version('tabpfn')}; pandas {version('pandas')}", flush=True)
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
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "customers.parquet"
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
        print(f"Fitting {offset} day(s) after customer creation.", flush=True)
        fitted = context.fit("tabpfn-rel-local", n_trials=0)
        print("Predicting for new customers with populated user_type.", flush=True)
        print(
            fitted.predict(
                PredictiveQuery(entities=[20, 21], at_timestamp="test_timestamp")
            )
        )


if __name__ == "__main__":
    main()
