"""Compare real prediction behavior in an original checkout and installed wheels.

Run with --baseline CHECKOUT --fixture DIRECTORY --output DIRECTORY. The fixture
contains binary_classification/task.yaml and regression/task.yaml, as produced by
the TabPFN-Rel generated-database example. Uses the local backend and its weights.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fastdfs
import numpy as np
import pandas as pd

import relarena

if os.environ.get("RELARENA_COMPARE_BASELINE") == "1":
    from relarena.featurization import dfs
    from relarena.userdb import PredictiveQuery, PredictiveQuerySpec
else:
    from relarena_core.featurization import dfs
    from relarena_core.userdb import PredictiveQuery, PredictiveQuerySpec


def frame_digest(frame: pd.DataFrame) -> str:
    """Hash values and dtypes to detect split or feature changes."""
    content = frame.to_json(date_unit="ns", orient="split")
    return hashlib.sha256((content + str(frame.dtypes)).encode()).hexdigest()


def snapshot(fixture: Path, output: Path, reuse_cache: Path | None = None) -> None:
    """Record splits, selected configurations, features and real predictions."""
    output.mkdir(parents=True, exist_ok=False)
    snapshots = []
    cache_root = reuse_cache or output
    cache_files = {
        path: path.stat().st_mtime_ns
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    if reuse_cache:
        fastdfs.compute_dfs_features = reject_dfs_computation
    for task_type in ("binary_classification", "regression"):
        directory = fixture / task_type
        query = PredictiveQuery(
            PredictiveQuerySpec.from_yaml(directory / "task.yaml", data_dir=directory),
            data_version="generated-v1",
        )
        splits = {}
        for phase, split in (
            ("inner", query._source.inner_split()),
            ("outer", query._source.outer_split()),
        ):
            splits[phase] = {
                "cutoff": str(split.cutoff),
                "database": {
                    name: frame_digest(table.df)
                    for name, table in sorted(split.db_state.table_dict.items())
                },
                "train": frame_digest(split.train_table.df),
                "evaluation": frame_digest(split.eval_table.df),
            }
        features = []
        original_type_columns = dfs.type_columns

        def capture(frame: pd.DataFrame, drop: set[str]) -> Any:
            typed, categoricals = original_type_columns(frame, drop)
            features.append(
                {
                    "input_dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
                    "typed_hash": frame_digest(typed),
                }
            )
            return typed, categoricals

        dfs.type_columns = capture
        try:
            query.fit("tabpfn-rel-local", n_trials=2, cache_dir=cache_root / task_type)
            first = query.predict()
            second = query.predict()
            third = query.predict()
        finally:
            dfs.type_columns = original_type_columns
        pd.testing.assert_frame_equal(second, third, rtol=1e-5, atol=1e-7)
        snapshots.append(
            {
                "task_type": task_type,
                "identity": asdict(query._identity),
                "splits": splits,
                "config": query.config,
                "trial_configs": [trial.config for trial in query.trials],
                "scores": [trial.val_score for trial in query.trials],
                "predictions": [first.y_pred.tolist(), second.y_pred.tolist()],
                "labels": query.compute_test_labels().to_json(date_unit="ns"),
                "features": features,
                "cache_paths": sorted(
                    str(p.relative_to(output)) for p in output.rglob("*.parquet")
                ),
                "cold_warm_max_difference": float(
                    np.max(np.abs(first.y_pred - second.y_pred))
                ),
            }
        )
    if reuse_cache:
        assert cache_files
        assert cache_files == {
            path: path.stat().st_mtime_ns
            for path in cache_root.rglob("*")
            if path.is_file()
        }, "Reusing baseline cache changed its files"
    (output / "snapshot.json").write_text(
        json.dumps(
            {
                "module": relarena.__file__,
                "warm_signature": str(
                    inspect.signature(PredictiveQuery.precompute_cache)
                ),
                "tasks": snapshots,
            },
            indent=2,
        )
        + "\n"
    )


def reject_dfs_computation(*args: Any, **kwargs: Any) -> None:
    """Fail if a cache reuse check tries to compute fresh DFS features."""
    raise AssertionError("Expected to reuse the original checkout's DFS artifacts")


def main() -> None:
    """Run both checkouts with identical dependencies and compare their output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reuse-cache", type=Path)
    args = parser.parse_args()
    if args.worker:
        snapshot(
            args.fixture.resolve(),
            args.output.resolve(),
            args.reuse_cache.resolve() if args.reuse_cache else None,
        )
        return
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=baseline, text=True
    ).strip()
    for name, cwd in (("baseline", baseline), ("candidate", output)):
        env = dict(os.environ, OMP_NUM_THREADS="1")
        env.pop("PYTHONPATH", None)
        env["RELARENA_COMPARE_BASELINE"] = "1" if name == "baseline" else "0"
        if name == "baseline":
            env["PYTHONPATH"] = str(baseline / "src")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--fixture",
            str(args.fixture.resolve()),
            "--output",
            str(output / name),
        ]
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=log, check=True
            )
    before = json.loads((output / "baseline/snapshot.json").read_text())
    after = json.loads((output / "candidate/snapshot.json").read_text())
    assert Path(before.pop("module")).is_relative_to(baseline / "src")
    assert not Path(after.pop("module")).is_relative_to(baseline)
    differences = []
    for old, new in zip(before["tasks"], after["tasks"], strict=True):
        differences.append(
            {
                "task_type": old["task_type"],
                "cold_warm_max_difference": new.pop("cold_warm_max_difference"),
            }
        )
        old.pop("cold_warm_max_difference")
        for key in ("predictions", "scores"):
            np.testing.assert_allclose(old.pop(key), new.pop(key), rtol=1e-5, atol=1e-7)
    assert before == after, "Splits, configuration, feature or cache snapshots differ"
    report = {
        "baseline_head": baseline_head,
        "passed": True,
        "cache_observations": differences,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
