# RelArena examples

Two runnable examples, answering two different questions. Both are run from the
repository root.

| Example | Question it answers | Needs |
|---|---|---|
| [`olist_seller_churn.py`](olist_seller_churn.py) | How do I run a model on **my own** database? | Kaggle account; no GPU by default |
| [`tabpfn_rel_caching.py`](tabpfn_rel_caching.py) | Why is the feature cache worth setting up? | RelBench download; GPU recommended |

Start with the Olist example if you are evaluating RelArena on your own data;
start with the caching one if you are setting up a sweep.

## `olist_seller_churn.py` — Relational Predictive Interface (RPI)

The full RPI path a user takes for their own data, on the Kaggle Olist Brazilian
E-Commerce dataset (7 tables of real e-commerce data): a task YAML references a
database YAML whose tables point straight at the raw CSVs, then fit, predict, and
evaluate on a held-out historical window. Predicts seller churn — among sellers
active in the last 30 days, which will have no order in the next 30 days?

Three files belong to this example:

- `olist_seller_churn.py` — the script
- `olist_seller_churn.yaml` — the task: label SQL, entity, split timestamps
- `olist_database.yaml` — the schema manifest the task YAML references, curating
  which raw CSV columns become features

```bash
uvx kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
uv sync --extra tabpfn-rel-api
uv run python -c "from tabpfn_client import init; init()"
OMP_NUM_THREADS=1 uv run --no-sync python examples/olist_seller_churn.py
```

That default runs through the hosted TabPFN API, so it needs no GPU. To run the
model locally instead (needs `uv sync --extra rdblearn`, GPU recommended):

```bash
OMP_NUM_THREADS=1 uv run --no-sync python examples/olist_seller_churn.py --backend local
```

Relational context pays off on this task: held-out ROC AUC is 0.50 for the global
constant baseline, 0.58 for lightgbm on entity-only features, 0.69 for the
per-entity constant baseline (each seller's own past churn rate), and 0.79 for TabPFN-Rel.

For the concepts behind the task YAML — how to write the label query, choose
split timestamps, and avoid leakage — see
[`docs/predictive-task.md`](../docs/predictive-task.md).

## `tabpfn_rel_caching.py` — what the feature cache buys

Fits one RelBench task (rel-f1 / driver-dnf) twice, once with no cache and once
against a store warmed up front, and checks the predictions are identical — the
cache only changes speed, never results. On that task it turns a roughly 409s
fit-and-predict into roughly 12s.

```bash
uv run --extra rdblearn python examples/tabpfn_rel_caching.py
```

The expensive step being cached is Deep Feature Synthesis, which runs on CPU. To
exercise the cache path without a GPU, skip the TabPFN forward pass:

```bash
RELARENA_EXAMPLE_SKIP_TFM=1 OMP_NUM_THREADS=1 \
    uv run --extra rdblearn python examples/tabpfn_rel_caching.py
```

See the feature-cache section of the [package README](../README.md) for how to
point a real sweep at a shared store.

## On macOS, prefix with `OMP_NUM_THREADS=1`

torch and lightgbm bundle separate libomp runtimes, and lightgbm segfaults if
torch loads first — a known macOS-only issue
([LightGBM#6595](https://github.com/microsoft/LightGBM/issues/6595)), not a bug
in relarena. Linux is unaffected.
