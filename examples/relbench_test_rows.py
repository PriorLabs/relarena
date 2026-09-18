"""Predict and evaluate the canonical test rows of a RelBench-v1 task.

Run from the repository root (downloads the selected dataset):
    OMP_NUM_THREADS=1 uv run --no-sync python examples/relbench_test_rows.py \
        --dataset rel-f1 --task driver-dnf

The label SQL selects the eligible entities at each test timestamp and computes
outcomes from the full database. Prediction uses the fitted context's frozen
database and each query's timestamp; labels are used only for evaluation.
"""

from __future__ import annotations

import argparse

import pandas as pd

from relarena.userdb import (
    PredictiveContext,
    PredictiveQuery,
    materialize_relbench,
    relbench_v1_spec,
)


def predict_test_rows(context: PredictiveContext, model: str) -> pd.DataFrame:
    """Fit once and align predictions with every labeled test row."""
    labels = context.compute_test_labels()
    fitted = context.fit(model, n_trials=0)
    batches = []
    for timestamp, entity_ids in context.group_test_entities(labels):
        query = PredictiveQuery(entities=entity_ids, at_timestamp=timestamp)
        batches.append(fitted.predict(query))
    if not batches:
        raise ValueError("The test cohort is empty.")
    scored = labels.merge(
        pd.concat(batches, ignore_index=True),
        on=[context.task.time_col, context.task.entity_col],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not scored["_merge"].eq("both").all():
        raise ValueError("Prediction rows must exactly match test label rows.")
    scored = scored.drop(columns="_merge")
    if scored[f"{context.task.target_col}_pred"].isna().any():
        raise ValueError("Predictions are missing rows from the test cohort.")
    return scored


def main() -> None:
    """Evaluate a registered model on a reference task's test cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="constant-global")
    parser.add_argument("--data-dir", default="data/relbench_v1")
    args = parser.parse_args()
    data_dir = materialize_relbench(args.dataset, f"{args.data_dir}/{args.dataset}")
    context = PredictiveContext(
        relbench_v1_spec(args.dataset, args.task, data_dir=str(data_dir))
    )
    scored = predict_test_rows(context, args.model)
    target = context.task.target_col
    for metric in context.task.metrics:
        print(
            f"{metric.__name__}: {metric(scored[target], scored[f'{target}_pred']):.4f}"
        )


if __name__ == "__main__":
    main()
