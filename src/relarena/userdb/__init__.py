"""Relational Predictive Interface (RPI) for user-defined tasks.

A user describes a prediction task by a windowed-aggregation SQL query that
produces `(time_col, entity_col, target_col)` label rows — the shape
RelBench's `EntityTask.make_table` emits — over a database supplied as CSV or
Parquet files plus YAML database and task specifications. `PredictiveQuerySpec`
bundles the whole
task into one object (loadable from YAML, or built in code from a
`DatabaseSpec` + `PredictiveTaskSpec`);
`PredictiveQuery(spec).fit(model).predict()` runs it end to end, and
`PredictiveQuery.compute_test_labels` materializes outcomes for historical test
windows when the source data cover their label horizon. The 21 RelBench-v1 entity
tasks ship as reference specs (`relbench_v1_spec`).
"""

from relarena.userdb.ingest import DatabaseSpec
from relarena.userdb.query import PredictiveQuery, PredictiveQuerySpec
from relarena.userdb.relbench_v1 import (
    materialize_relbench,
    relbench_v1_spec,
    relbench_v1_tasks,
)
from relarena.userdb.spec import PredictiveTaskSpec

__all__ = [
    "DatabaseSpec",
    "PredictiveQuery",
    "PredictiveQuerySpec",
    "PredictiveTaskSpec",
    "materialize_relbench",
    "relbench_v1_spec",
    "relbench_v1_tasks",
]
