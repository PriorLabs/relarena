"""The 21 RelBench-v1 entity tasks as reference specs + a data helper.

One subfolder per dataset: a shared `db.yaml` (the database schema - primary keys /
time columns / foreign keys) plus one `<task>.yaml` per task (the split timestamps,
windowed-aggregation label SQL, and prediction target, each referencing `db.yaml`).
Pair a task with parquet from `materialize_relbench` and run it:

    from relarena.userdb import (
        PredictiveContext, PredictiveQuery, materialize_relbench, relbench_v1_spec,
    )

    materialize_relbench("rel-f1", "data/rel-f1")
    spec = relbench_v1_spec("rel-f1", "driver-dnf", data_dir="data/rel-f1")
    fitted = PredictiveContext(spec).fit(model="tabpfn-rel-client-latest")
    query = PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    preds = fitted.predict(query)

The specs reproduce RelBench's own `make_table` output byte-for-byte (verified
across all 21 tasks), including its split timestamps. The materialized tables are
not restricted to RelBench's test cutoff: copy a reference task YAML and change its
split timestamps to define a different RPI problem over the same source data.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from relbench.datasets import get_dataset

from relarena_core.userdb.query import PredictiveQuerySpec

_SPEC_DIR = files(__name__)


def materialize_relbench(dataset: str, dest: str, *, download: bool = True) -> Path:
    """Write a RelBench dataset's tables to parquet under `dest` (one per table).

    A spec loaded with `data_dir=dest` finds these tables. The export holds the
    full database, including rows after RelBench's original test cutoff; the task
    YAML chooses the val/test timestamps.

    Args:
        dataset: RelBench dataset name, e.g. `"rel-f1"`.
        dest: Output directory for `<table>.parquet` files.
        download: Fetch RelBench's prepared database; `False` builds it locally.

    Returns:
        `dest` as a `Path`.
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

    The bundled spec keeps RelBench's original val/test timestamps to reproduce
    the benchmark; copy and edit the task YAML to choose other cutoffs.

    Args:
        dataset: RelBench dataset name, e.g. `"rel-f1"`.
        task: Task name, e.g. `"driver-dnf"`; `relbench_v1_tasks()` lists them.
        data_dir: Directory written by `materialize_relbench`. Without it only the
            schema and task fields are usable, not a `PredictiveContext`.
    """
    path = _SPEC_DIR / dataset / f"{task}.yaml"
    return PredictiveQuerySpec.from_yaml(str(path), data_dir=data_dir)
