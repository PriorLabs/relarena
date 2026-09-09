"""Exercise real TabPFN inference over the generated relational example.

From this checkout, run ``python -m workflows.check_inference --output PATH``.
The local backend needs model weights. The client backend needs authentication
and consumes service quota. Only generated customer/event data are used.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from examples.tiny_database import write_database
from tabpfn_rel import PredictiveQuery, PredictiveQuerySpec


def main() -> None:
    """Fit classification and regression models, then check warm-cache predictions and tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "client"], default="local")
    parser.add_argument("--n-trials", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for task_type in ("binary_classification", "regression"):
        directory = output / task_type
        task_path = write_database(directory, task_type)
        spec = PredictiveQuerySpec.from_yaml(str(task_path), data_dir=str(directory))
        query = PredictiveQuery(spec, data_version="generated-v1").fit(
            f"tabpfn-rel-{args.backend}",
            n_trials=args.n_trials,
            cache_dir=directory / "cache",
        )
        cold_predictions = query.predict()
        predictions = query.predict()
        repeated = query.predict()
        assert sorted(predictions["customer_id"]) == ["a", "b", "c", "d"]
        assert np.isfinite(predictions["y_pred"]).all()
        pd.testing.assert_frame_equal(predictions, repeated, rtol=1e-5, atol=1e-7)
        if task_type == "binary_classification":
            assert predictions["y_pred"].between(0, 1).all()
        labels = query.compute_test_labels()
        assert len(labels) == len(predictions) == 4
        assert list((directory / "cache").rglob("*.parquet"))
        if args.n_trials:
            assert query.trials and all(
                t.ok and np.isfinite(t.val_score) for t in query.trials
            )
        predictions.to_csv(directory / "predictions.csv", index=False)
        labels.to_csv(directory / "labels.csv", index=False)
        result = {
            "task_type": task_type,
            "backend": args.backend,
            "backend_version": version(
                "tabpfn" if args.backend == "local" else "tabpfn-client"
            ),
            "config": query.config,
            "validation_scores": [t.val_score for t in query.trials or []],
            "rows": len(predictions),
            "cold_warm_max_difference": float(
                np.max(np.abs(cold_predictions["y_pred"] - predictions["y_pred"]))
            ),
            "passed": True,
        }
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            f"{task_type}: passed with {len(predictions)} real predictions", flush=True
        )


if __name__ == "__main__":
    main()
