"""A RelBench `EntityTask` whose label table is produced by a user SQL query."""

from __future__ import annotations

import duckdb
import pandas as pd
from relbench.base import Database, Dataset, EntityTask, Table, TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, r2, rmse, roc_auc

from relarena.userdb.spec import PredictiveTaskSpec

#: RelBench's standard metric lists per entity task type (primary metric last).
_METRICS_BY_TASK_TYPE = {
    TaskType.BINARY_CLASSIFICATION: [average_precision, accuracy, f1, roc_auc],
    TaskType.REGRESSION: [r2, mae, rmse],
}


class UserEntityTask(EntityTask):
    """An `EntityTask` whose `make_table` runs the spec's windowed SQL query.

    Everything else — the train/val/test anchor-time grid, `upto` censoring,
    test-label masking and dangling-entity filtering — is inherited unchanged
    from `relbench.base.EntityTask`, so the produced tables match a
    hand-written RelBench task whenever the query computes the same labels.
    """

    def __init__(self, dataset: Dataset, spec: PredictiveTaskSpec) -> None:
        """Build the task from a RelBench `dataset` and a `PredictiveTaskSpec`."""
        self.entity_table = spec.entity_table
        self.entity_col = spec.entity_col
        self.time_col = spec.time_col
        self.target_col = spec.target_col
        self.task_type = spec.task_type
        self.timedelta = spec.timedelta
        self.num_eval_timestamps = spec.num_eval_timestamps
        self.metrics = _METRICS_BY_TASK_TYPE[spec.task_type]
        self._query = spec.query
        super().__init__(dataset, cache_dir=None)

    def make_table(
        self,
        db: Database,
        timestamps: "pd.Series[pd.Timestamp]",
    ) -> Table:
        """Run the spec's SQL over `db` at the given anchor `timestamps`."""
        # 'timestamp_df' is reserved: it holds the anchor times the label query
        # joins against. A source table of that name would overwrite it and the
        # labels would be computed at the wrong timestamps.
        if "timestamp_df" in db.table_dict:
            raise ValueError(
                "'timestamp_df' is a reserved table name - the label query uses it "
                "for the anchor timestamps. Rename the source table called "
                "'timestamp_df' to something else."
            )
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        con = duckdb.connect()
        try:
            con.register("timestamp_df", timestamp_df)
            for name, table in db.table_dict.items():
                con.register(name, table.df)
            # .replace, not .format: user SQL may contain literal braces (DuckDB
            # struct/MAP/JSON literals), which .format would misparse as fields.
            query = self._query.replace("{timedelta}", str(self.timedelta))
            target_df = con.sql(query).df()
        finally:
            con.close()

        # Enforce the three-column contract: any extra column would silently
        # become a model input feature (leaked from the forward label window).
        expected = {self.time_col, self.entity_col, self.target_col}
        if set(target_df.columns) != expected:
            raise ValueError(
                f"Label query must return exactly {sorted(expected)} "
                f"(time_col, entity_col, target_col); got {list(target_df.columns)}."
            )

        key = [self.time_col, self.entity_col]
        dups = int(target_df.duplicated(key).sum())
        if dups:
            raise ValueError(
                f"Label query returned {dups} duplicate ({self.time_col}, "
                f"{self.entity_col}) rows; each entity may appear at most once per "
                f"anchor timestamp."
            )

        # Deterministic order: evaluate aligns preds to labels positionally and the
        # table is recomputed uncached, so sort by the unique (time, entity) key.
        target_df = target_df.sort_values(key).reset_index(drop=True)

        return Table(
            df=target_df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
