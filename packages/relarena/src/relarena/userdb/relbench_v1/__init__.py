"""The 21 RelBench-v1 entity tasks as reference specs + a data helper.

One subfolder per dataset: a shared `db.yaml` (the database schema - primary keys /
time columns / foreign keys) plus one `<task>.yaml` per task (the split timestamps,
windowed-aggregation label SQL, and prediction target, each referencing `db.yaml`).
Pair a task with parquet from `materialize_relbench` and run it:

    from relarena.userdb import PredictiveQuery, materialize_relbench, relbench_v1_spec

    materialize_relbench("rel-f1", "data/rel-f1")
    spec = relbench_v1_spec("rel-f1", "driver-dnf", data_dir="data/rel-f1")
    preds = PredictiveQuery(spec).fit(model="tabpfn-rel-client").predict()

The specs reproduce RelBench's own `make_table` output byte-for-byte (verified
across all 21 tasks), including its split timestamps. The materialized tables are
not restricted to RelBench's test cutoff: copy a reference task YAML and change its
split timestamps to define a different RPI problem over the same source data.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from relbench.datasets import get_dataset

from relarena.userdb.query import PredictiveQuerySpec

_SPEC_DIR = files(__name__)


def materialize_relbench(dataset: str, dest: str, *, download: bool = True) -> Path:
    """Write a RelBench dataset's tables to parquet under `dest` (one per table).

    The small data-only helper the specs expect: it lays down
    `<dest>/<table>.parquet`, so a spec loaded with `data_dir=<dest>` finds its
    tables. It exports the full database, including rows after RelBench's original
    test cutoff. The RPI task YAML independently chooses the val/test timestamps
    used to split and freeze those tables. Returns `dest`.
    """
    # Dump get_db's reindexed, time-sorted tables, not raw make_db output: userdb
    # ingest re-sorts them stably to reproduce RelBench's entity ids (see
    # _reindex_stable), which matches native only because they arrive time-sorted.
    # Raw tables would stay deterministic but drop the match with native's ids.
    db = get_dataset(dataset, download=download).get_db(upto_test_timestamp=False)
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in db.table_dict.items():
        table.df.to_parquet(out / f"{name}.parquet", index=False)
    return out


def relbench_v1_tasks() -> list[tuple[str, str]]:
    """List the available `(dataset, task)` reference specs.

    One subfolder per dataset holds its shared `db.yaml` plus one YAML per task.
    """
    pairs = []
    for folder in _SPEC_DIR.iterdir():
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.name.endswith(".yaml") and p.name != "db.yaml":
                pairs.append((folder.name, p.name[: -len(".yaml")]))
    return sorted(pairs)


def relbench_v1_spec(
    dataset: str, task: str, *, data_dir: str | None = None
) -> PredictiveQuerySpec:
    """Load the reference `PredictiveQuerySpec` for a RelBench-v1 task.

    The bundled spec deliberately retains RelBench's original val/test timestamps
    to reproduce the benchmark. They are properties of this reference task, not a
    restriction imposed by `materialize_relbench`; copy and edit the task YAML to
    choose different cutoffs over the same materialized data.

    `data_dir` is where the parquet lives (see `materialize_relbench`). If omitted,
    only the schema/task fields are usable (e.g. `.task`), not a `PredictiveQuery`
    run.
    """
    path = _SPEC_DIR / dataset / f"{task}.yaml"
    return PredictiveQuerySpec.from_yaml(str(path), data_dir=data_dir)
