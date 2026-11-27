# relflow

The relflow 0.5.3 adapter, `relflow-2026-11-25` (`relflow-latest`), derives its
schema from the supplied temporal split. A single frozen recipe covers each range of
original training-anchor counts. The same configuration carries through the
train-plus-validation refit. [relflow 0.5.3 source](https://github.com/relflow/relflow/tree/v0.5.3)

## Running relflow 0.5.3

Install the `relflow` extra with Python 3.12. To run the standard benchmark at
seed zero on one GPU:

```bash
uv sync --locked --python 3.12.10 --all-packages --extra relflow
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  POLARS_MAX_THREADS=1 TOKENIZERS_PARALLELISM=false \
  uv run --no-sync relarena --model relflow-2026-11-25 --seed 0 --n-trials 0 \
    --datasets rel-f1 --tasks driver-dnf --output relflow_driver_dnf.csv
```

Omit the dataset and task filters to evaluate all 21 tasks. The standard runner
fits its single default on the inner split and refits on the full outer training
table for test predictions. Use a fresh process for each seed and set `PYTHONHASHSEED` to the same
value as `--seed`. The adapter uses deterministic kernels and one PyTorch
inter-op thread per process. GPU architecture and the PyTorch/CUDA build can affect numerical results. Development
runs used Python 3.12.10, PyTorch 2.13.0 with CUDA 13.0 and relflow 0.5.3.

## Schema and training in relflow 0.5.3

Each anchor can see its entity’s earlier outcomes once the forecast horizon has
elapsed. The network receives each outcome and its age, plus the arithmetic
mean of all visible outcomes. Generic foreign-key paths supply peer counts and
means that exclude the current entity. Direct timed relations provide event
counts and means. Fitting uses only labels supplied to the model. Predictions
freeze outcome availability at the phase boundary.

Integral regression labels select an Ordinal target, and other numeric labels
select Number. The nullable target policy applies to nonnegative labels with
at least one positive and at least 5% zeros. It encodes zero as null during
training and returns zero at a corrected null probability of 0.5 or above.
The SMALL preset enables this policy for Ordinal. The shared MEDIUM/LARGE
preset enables it for both Ordinal and Number. Text uses at most 128 tokens from the
same pinned MiniLM encoder as the earlier entry.

| Original training anchors | Preset | Batch size | Pass equivalents | Root / branch crossing |
| --- | --- | ---: | ---: | --- |
| Below 20,000 | B26 | 32 | 8 | On / on |
| 20,000 and above | M74 | 256 | 2 | Off / off |

The final range shares one architecture and one schema budget, including on the
largest tasks. Both presets use fixed batch sizes, zero loader workers, AdamW,
a cosine learning-rate schedule and EMA. For the largest range the outer-fit
guard checks the projected total during the first pass, targeting 16.8 hours
and enforcing a 24-hour fit-plus-prediction budget. Full training must finish
to return a score. The complete settings are in
[the relflow 0.5.3 recipes](../../packages/relarena/src/relarena/models/relflow/recipes-0.5.3.json).

## relflow 0.5.1

To run the November 4 adapter and its original recipes with relflow 0.5.1,
select the dated extra and model.

```bash
uv sync --locked --python 3.12.10 --all-packages --extra relflow-2026-11-04
uv run --no-sync relarena --model relflow-2026-11-04 --seed 0 --n-trials 0 \
  --datasets rel-f1 --tasks driver-dnf --output relflow_0_5_1_driver_dnf.csv
```

Both adapters require Python 3.12 and DuckDB 1.5 or newer. The KurveRSC extra
requires a separate environment because of its NumPy constraint.

### Model and training

The original inner training-anchor count selects the tier and remains fixed
during the train+validation refit. Fields, datatypes and bounded join paths are
derived from keys and validation-censored column measurements. The model consumes
raw values and timestamp ages, with per-anchor censoring of histories and timed
lookups.

| Setting | SMALL | MEDIUM | LARGE |
| --- | ---: | ---: | ---: |
| Inner training anchors | <20,000 | 20,000–499,999 | ≥500,000 |
| Width / root layers / heads | 128 / 2 / 4 | 64 / 2 / 8 | 64 / 2 / 8 |
| Branch layers / heads / summary vectors | 1 / 4 / 4 | 2 / 4 / 1 | 2 / 4 / 1 |
| Model outer / inner history caps | 128 / 32 | 64 / 32 | 32 / 16 |
| Training batch size | 64 | 512 | 512 |
| Loader workers on CUDA | 0 | 0 | 2 |
| Shuffle buffer rows | 16,384 | 16,384 | 2,048 |
| DuckDB memory / spill limits | 16 / 64 GB | 16 / 128 GB | 32 / 256 GB |

The complete settings live in
[`recipes.json`](../../packages/relarena/src/relarena/models/relflow/recipes.json).
All tiers use eight pass-equivalents, AdamW at a constant learning rate,
gradient clipping at 0.25, zero neural dropout and EMA with decay 0.995.
Regression optimizes MAE. Training randomly crops contiguous history windows;
prediction uses deterministic tails. Text fields use frozen
`sentence-transformers/all-MiniLM-L6-v2` weights at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

The final refit intentionally places shuffled training anchors before
validation-period anchors on every source pass. The bounded shuffle buffer
retains this recency bias, so the final EMA weights emphasize validation-period
rows. Those anchors retain histories frozen at the validation boundary;
prediction uses the harness-supplied evaluation database.

### Runtime and resources

Preparation, text encoding and training run inside `fit`; no separate cache
warmer is required. Default-only benchmark timing includes the final refit and
prediction; the diagnostic inner fit is recorded separately. The runtime ceiling
is 24 hours on the largest tasks, including preparation. See the
[runtime policy](../tuning-regime.md).

GPU execution is recommended for the full suite. DuckDB limits in the table
bound its own memory and spill use; they are not measured peak host or GPU
memory for the complete process. CPU loaders always use zero workers. An
out-of-memory error fails the run without reducing its batch or training budget.
Maintainers reproduce submissions on their benchmark hardware and add
`baseline_results` after the rerun.
