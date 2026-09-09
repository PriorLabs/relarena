"""Predict on a generated two-table database with local or hosted TabPFN.

From this package directory, run
``uv run --package tabpfn-rel --extra local python examples/tiny_database.py``.
For the hosted backend, install the api extra and pass ``--backend client``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from tabpfn_rel import PredictiveQuery, PredictiveQuerySpec


def write_database(directory: Path, task_type: str = "binary_classification") -> Path:
    """Write a small relational dataset and a forward-looking prediction task."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "age": [22, 34, 46, 58],
            "description": ["first customer", "second customer", "third", "fourth"],
        }
    ).to_parquet(directory / "customers.parquet")
    dates = pd.date_range("2004-01-15", "2005-06-15", freq="30D")
    events = [
        {"customer_id": customer, "date": date, "amount": float(10 + i)}
        for i, date in enumerate(dates)
        for j, customer in enumerate(["a", "b", "c", "d"])
        if (i + j) % 2 == 0
    ]
    pd.DataFrame(events).assign(event_id=range(len(events))).to_parquet(
        directory / "events.parquet"
    )
    (directory / "database.yaml").write_text(
        "customers:\n  pkey: customer_id\n"
        "events:\n  pkey: event_id\n  time_col: date\n"
        "  fkeys:\n    customer_id: customers\n"
    )
    aggregate = "CAST(COUNT(e.event_id) > 0 AS INTEGER)"
    if task_type == "regression":
        aggregate = "COALESCE(SUM(e.amount), 0)"
    task = directory / "task.yaml"
    task.write_text(
        "database: database.yaml\nentity_table: customers\n"
        "entity_col: customer_id\ntime_col: date\ntarget_col: y\n"
        f"task_type: {task_type}\ntimedelta: 30 days\n"
        "val_timestamp: '2004-10-01'\ntest_timestamp: '2004-12-01'\n"
        "query: |\n"
        f"  SELECT t.timestamp AS date, c.customer_id, {aggregate} AS y\n"
        "  FROM timestamp_df t CROSS JOIN customers c\n"
        "  LEFT JOIN events e ON e.customer_id = c.customer_id\n"
        "    AND e.date > t.timestamp\n"
        "    AND e.date <= t.timestamp + INTERVAL '{timedelta}'\n"
        "  GROUP BY t.timestamp, c.customer_id\n"
    )
    return task


def main() -> None:
    """Generate data, fit the selected model and print its predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "client"], default="local")
    parser.add_argument("--n-trials", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        task = write_database(Path(tmp))
        spec = PredictiveQuerySpec.from_yaml(str(task), data_dir=tmp)
        query = PredictiveQuery(spec, data_version="example-v1").fit(
            f"tabpfn-rel-{args.backend}", n_trials=args.n_trials
        )
        print(query.predict().to_string(index=False))


if __name__ == "__main__":
    main()
