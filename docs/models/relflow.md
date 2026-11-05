# RelFlow

[RelFlow](https://github.com/relflow/relflow) builds a neural model over typed,
nested records. This adapter derives fields and joins from the supplied
relational database and trains one fixed recipe per training-set size tier.

## Installation

The optional backend requires Python 3.12:

```bash
uv sync --python 3.12 --all-packages --extra relflow
```

For CPU execution, add `--group cpu`. The `relflow` extra cannot be combined
with KurveRSC 0.1.1, which requires NumPy 1; RelFlow requires NumPy 2.

## Running RelFlow

Run one task through the RelArena CLI:

```bash
OMP_NUM_THREADS=1 uv run --all-packages --extra relflow relarena \
    --model relflow-2026-11-04 \
    --datasets rel-f1 --tasks driver-dnf \
    --output relflow_driver_dnf.csv
```

Omit `--datasets` and `--tasks` to run all 21 RelBench v1 entity tasks.
The registered search space contains one configuration per tier, so
`--n-trials` does not add candidate configurations.

## Model and training

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
| DuckDB memory / spill limits | 4 / 64 GB | 16 / 128 GB | 32 / 256 GB |

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

## Runtime and resources

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
