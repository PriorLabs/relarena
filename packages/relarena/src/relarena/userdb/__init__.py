"""Relational Predictive Interface (RPI) for user-defined tasks.

A user describes a prediction task by a windowed-aggregation SQL query that
produces `(time_col, entity_col, target_col)` label rows — the shape
RelBench's `EntityTask.make_table` emits — over a database supplied as CSV or
Parquet files plus YAML database and task specifications. `PredictiveQuerySpec`
bundles the whole
task into one object (loadable from YAML, or built in code from a
`DatabaseSpec` + `PredictiveTaskSpec`);
`PredictiveContext(spec).fit(model)` returns a fitted predictor. Pass an explicit
`PredictiveQuery` to its `predict` method. `PredictiveContext.compute_test_labels`
materializes outcomes for historical test windows.
"""

from relarena.userdb.relbench_v1 import (
    materialize_relbench,
    relbench_v1_spec,
    relbench_v1_tasks,
)
from relarena_core.userdb.ingest import DatabaseSpec
from relarena_core.userdb.query import (
    FittedPredictor,
    PredictiveContext,
    PredictiveQuery,
    PredictiveQuerySpec,
)
from relarena_core.userdb.spec import PredictiveTaskSpec

__all__ = [
    "DatabaseSpec",
    "PredictiveContext",
    "FittedPredictor",
    "PredictiveQuery",
    "PredictiveQuerySpec",
    "PredictiveTaskSpec",
    "materialize_relbench",
    "relbench_v1_spec",
    "relbench_v1_tasks",
]
