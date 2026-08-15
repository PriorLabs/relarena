r"""Verify the RelBench-v1 reference specs reproduce native RelBench task tables.

Slow: downloads + materializes each RelBench dataset, rebuilds every task through
the userdb interface, and diffs its train/val/test label tables against RelBench's
own `get_task`. Run it deliberately - after editing a spec or bumping RelBench, or
just to see the reproduction hold - not on every load.

    python workflows/verify_relbench_v1.py                     # all 21 tasks
    python workflows/verify_relbench_v1.py --datasets rel-f1   # one dataset (small)
    python workflows/verify_relbench_v1.py --tasks driver-dnf  # one task

Exits non-zero if any split differs from RelBench.
"""

from __future__ import annotations

import argparse

import pandas as pd
from relbench.tasks import get_task

from relarena.dataset import drop_noncanonical_task_columns
from relarena.userdb import (
    materialize_relbench,
    relbench_v1_spec,
    relbench_v1_tasks,
)
from relarena.userdb.ingest import build_dataset
from relarena.userdb.task import UserEntityTask


def _split_matches(want: pd.DataFrame, got: pd.DataFrame, keys: list[str]) -> bool:
    """True if the two label tables are equal up to row order (compared on want's cols)."""
    cols = list(want.columns)
    want = want.sort_values(keys).reset_index(drop=True)[cols]
    got = got.sort_values(keys).reset_index(drop=True)[cols]
    try:
        pd.testing.assert_frame_equal(want, got, check_dtype=False, check_like=True)
        return True
    except AssertionError:
        return False


def main(argv: list[str] | None = None) -> int:
    """Diff each v1 task's split tables: userdb interface vs native RelBench."""
    p = argparse.ArgumentParser(prog="verify_relbench_v1", description=__doc__)
    p.add_argument("--datasets", nargs="*", default=None, help="restrict to these")
    p.add_argument("--tasks", nargs="*", default=None, help="restrict to these tasks")
    p.add_argument("--data-dir", default="data/relbench_v1", help="materialize root")
    args = p.parse_args(argv)

    specs = relbench_v1_tasks()
    if args.datasets:
        specs = [(d, t) for d, t in specs if d in set(args.datasets)]
    if args.tasks:
        specs = [(d, t) for d, t in specs if t in set(args.tasks)]

    materialized: dict[str, str] = {}
    failures: list[str] = []
    for dataset, task in specs:
        if dataset not in materialized:
            materialized[dataset] = str(
                materialize_relbench(dataset, f"{args.data_dir}/{dataset}")
            )
        native = get_task(dataset, task, download=True)
        spec = relbench_v1_spec(dataset, task, data_dir=materialized[dataset])
        ds = build_dataset(
            spec.database,
            val_timestamp=spec.task.val_timestamp,
            test_timestamp=spec.task.test_timestamp,
        )
        ours = UserEntityTask(ds, spec.task)
        keys = [spec.task.time_col, spec.task.entity_col]
        for split in ("train", "val", "test"):
            want = drop_noncanonical_task_columns(
                native, native.get_table(split, mask_input_cols=False), dataset
            ).df
            got = drop_noncanonical_task_columns(
                ours, ours.get_table(split, mask_input_cols=False), dataset
            ).df
            ok = _split_matches(want, got, keys)
            note = (
                f"({len(want)} rows)" if ok else f"(want {len(want)}, got {len(got)})"
            )
            print(f"{dataset}.{task} [{split}]: {'MATCH' if ok else 'DIFF'} {note}")
            if not ok:
                failures.append(f"{dataset}.{task}[{split}]")

    print(f"\n{len(specs)} task(s) checked; {len(failures)} split mismatch(es).")
    if failures:
        print("Mismatches:", ", ".join(failures))
        return 1
    print("All splits reproduce RelBench exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
