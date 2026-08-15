"""End-to-end: predict seller churn with TabPFN-Rel over a real bring-your-own DB.

Runs on the Kaggle Olist Brazilian E-Commerce dataset, walking the full path a user
takes for their own data: a task YAML (olist_seller_churn.yaml) references a database
YAML (olist_database.yaml) whose tables point straight at the raw CSVs - each table's
`columns` curate which fields become features (and drop post-purchase leak columns) -
then fit -> predict -> evaluate on a held-out historical window. Only `order_items`
is pre-processed, since it needs a derived event-time column. The task
(examples/olist_seller_churn.yaml):
entity = seller, binary label = among sellers active in the last 30 days, will they
have NO order in the next 30 days?

On this task the relational context pays off - cross-table order/review history
from the 7-table schema lifts held-out ROC-AUC above every baseline,
including each seller's own past churn rate: constant-global 0.50, lightgbm
(entity-only) 0.58, constant-per-entity 0.69, TabPFN-Rel 0.79.

Setup (needs a Kaggle account + ~/.kaggle/kaggle.json), from the repository root:

    uvx kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
    uv sync --extra tabpfn-rel-api
    uv run python -c "from tabpfn_client import init; init()"
    OMP_NUM_THREADS=1 uv run --no-sync python examples/olist_seller_churn.py

The default uses the hosted TabPFN API. To run the model locally instead:

    uv sync --extra rdblearn
    OMP_NUM_THREADS=1 uv run --no-sync python examples/olist_seller_churn.py \
        --backend local
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from relarena.userdb import PredictiveQuery, PredictiveQuerySpec


def prepare_olist_data(csv_dir: str) -> Path:
    """Derive `order_items`' event-time column; every other table is used raw.

    `order_items` has no trustworthy native timestamp (its `shipping_limit_date` is
    unreliable) and the temporal machinery needs one to censor on, so join the
    order's purchase time onto it as `purchase_ts`. The other tables are read
    straight from the raw CSVs and curated via the spec's per-table `columns`
    (including dropping `orders`' post-purchase leak columns), so this is the only
    table that needs code. Writes order_items.csv into `csv_dir` and returns it.
    """
    csv = Path(csv_dir)
    orders = pd.read_csv(
        csv / "olist_orders_dataset.csv",
        usecols=["order_id", "order_purchase_timestamp"],
    )
    items = pd.read_csv(csv / "olist_order_items_dataset.csv").merge(
        orders, on="order_id", how="left"
    )
    items.rename(columns={"order_purchase_timestamp": "purchase_ts"}).to_csv(
        csv / "order_items.csv", index=False
    )
    return csv


def fit_predict_and_evaluate(
    spec: PredictiveQuerySpec,
    model: str,
    *,
    n_trials: int = 10,
) -> None:
    """Fit `model`, predict at the held-out test anchor, and evaluate.

    `fit` tunes the model's search space on the inner split (train->val), selects
    the best config by validation score, and refits it on the outer split.
    `predict` scores label-less rows without exposing the forward label window to
    the model. Because Olist contains the complete window after its historical test
    timestamp, `compute_test_labels` can materialize those outcomes for evaluation.
    The model is a run-time choice, not part of the spec, so swap it here to compare
    against constant / lightgbm baselines.
    """
    pq = PredictiveQuery(spec).fit(model, n_trials=n_trials, seed=0)
    preds = pq.predict()
    labels = pq.compute_test_labels()
    scored = labels.merge(
        preds,
        on=[pq.task.time_col, pq.task.entity_col],
        how="left",
        validate="one_to_one",
    )
    if scored[f"{pq.task.target_col}_pred"].isna().any():
        raise RuntimeError("Predictions are missing rows from the test cohort.")
    roc_auc = roc_auc_score(
        scored[pq.task.target_col], scored[f"{pq.task.target_col}_pred"]
    )
    print(preds.head())
    print(f"Test ROC-AUC: {roc_auc:.3f}")


def parse_args() -> argparse.Namespace:
    """Parse the example's local-versus-hosted inference choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("local", "client"),
        default="client",
        help="Hosted TabPFN API (default), or local TabPFN without text features.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Tuning fits (default: 10 locally, 0 for the hosted API).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/olist",
        help="Directory containing the downloaded Olist CSV files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    here = Path(__file__).parent
    data_dir = prepare_olist_data(args.data_dir)
    spec = PredictiveQuerySpec.from_yaml(
        str(here / "olist_seller_churn.yaml"), data_dir=str(data_dir)
    )
    model = {
        "local": "tabpfn-rel-local",
        "client": "tabpfn-rel-client",
    }[args.backend]
    n_trials = (
        args.n_trials
        if args.n_trials is not None
        else (0 if args.backend == "client" else 10)
    )
    fit_predict_and_evaluate(spec, model=model, n_trials=n_trials)
