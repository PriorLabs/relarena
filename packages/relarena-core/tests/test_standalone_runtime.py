"""Tuned predictive queries with a supplied model and no benchmark integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, EntityTask, Table

from relarena_core import RelArenaModel, registry
from relarena_core.search_space import SearchSpace
from relarena_core.userdb import PredictiveQuery, PredictiveQuerySpec
from relarena_core.userdb import query as query_module


@pytest.mark.parametrize("refit_full", [False, True])
def test_tuning_final_fit_and_original_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refit_full: bool
) -> None:
    customers = pd.DataFrame({"customer_id": ["a", "b", "c", "d"]})
    dates = pd.date_range("2004-01-15", "2005-06-15", freq="30D")
    events = pd.DataFrame(
        [
            {"customer_id": customer, "date": date}
            for i, date in enumerate(dates)
            for j, customer in enumerate(customers.customer_id)
            if (i + j) % 2 == 0
        ]
    )
    customers.to_parquet(tmp_path / "customers.parquet")
    events.assign(event_id=range(len(events))).to_parquet(tmp_path / "events.parquet")
    (tmp_path / "database.yaml").write_text(
        "customers:\n  pkey: customer_id\n"
        "events:\n  pkey: event_id\n  time_col: date\n"
        "  fkeys:\n    customer_id: customers\n"
    )
    (tmp_path / "task.yaml").write_text("""database: database.yaml
entity_table: customers
entity_col: customer_id
time_col: date
target_col: y
task_type: binary_classification
timedelta: 30 days
val_timestamp: '2004-10-01'
test_timestamp: '2004-12-01'
query: |
  SELECT t.timestamp AS date, c.customer_id,
    CAST(COUNT(e.event_id) > 0 AS INTEGER) AS y
  FROM timestamp_df t CROSS JOIN customers c
  LEFT JOIN events e ON e.customer_id = c.customer_id
    AND e.date > t.timestamp
    AND e.date <= t.timestamp + INTERVAL '{timedelta}'
  GROUP BY t.timestamp, c.customer_id
""")
    fits = []

    class SuppliedModel(RelArenaModel):
        name = "test-standalone"
        refit_on_full_data = refit_full

        def fit(
            self,
            task: EntityTask,
            db: Database,
            train_table: Table,
            val_table: Table | None,
            *,
            seed: int,
            time_limit: float | None = None,
        ) -> None:
            if self.config["fail"]:
                raise ValueError("Deliberate failed tuning candidate")
            self.mean = float(train_table.df[task.target_col].mean())
            fits.append(
                (
                    len(train_table.df),
                    val_table is None,
                    db.table_dict["events"].df.date.max(),
                )
            )

        def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
            return np.full(len(table.df), self.mean)

    monkeypatch.setattr(query_module, "discover_models", lambda: None)
    monkeypatch.setattr(registry, "_entries", {})
    registry.register(
        SuppliedModel,
        SearchSpace(
            default_overrides={"fail": False},
            fixed_grid=[{"fail": True}, {"fail": False}],
        ),
    )
    spec = PredictiveQuerySpec.from_yaml(tmp_path / "task.yaml", data_dir=tmp_path)
    query = PredictiveQuery(spec, data_version="fixture-v1")
    inner, outer = query._source.inner_split(), query._source.outer_split()
    query.fit(SuppliedModel.name, n_trials=2)
    assert query.config == {"fail": False}
    assert [trial.ok for trial in query.trials] == [False, True]
    assert fits[0][0] == len(inner.train_table.df)
    assert fits[0][2] <= inner.cutoff
    expected_rows = len(outer.train_table.df)
    if refit_full:
        expected_rows += len(outer.val_table.df)
    assert fits[-1][:2] == (expected_rows, refit_full)
    assert fits[-1][2] <= outer.cutoff
    predictions = query.predict()
    assert sorted(predictions.customer_id) == ["a", "b", "c", "d"]
    assert np.isfinite(predictions.y_pred).all()
    assert len(query.compute_test_labels()) == 4
