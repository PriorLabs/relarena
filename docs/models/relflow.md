# RelFlow

RelFlow builds supervised neural models from schemas of scalar fields and repeated relational
histories. The `relflow-2026-10-21` system (`relflow-latest` alias) includes one recipe for each
of the 21 entity tasks, selected during development using validation data.

## Setup

Use uv **0.11.15 or newer**. RelFlow needs Arrow >=21, which conflicts with the default
development group's fastdfs dependency. Install its isolated environment:

```bash
uv sync --no-default-groups --all-packages --extra relflow --group relflow-dev --group cpu
```

Omit `--group cpu` for CUDA execution. A CUDA GPU is recommended for real benchmarks.

```bash
OMP_NUM_THREADS=1 uv run --no-default-groups --all-packages --extra relflow \
    --group relflow-dev --group cpu pytest packages/relarena/tests/models/relflow
```

Use the standard CLI to run one task; omit dataset/task filters for all 21:

```bash
uv run --no-default-groups --all-packages --extra relflow --group relflow-dev relarena \
    --model relflow-latest --datasets rel-f1 --tasks driver-dnf --output relflow.csv
```

## Architecture and protocol

Schemas describe scalar fields, per-field text limits, and bounded relational histories.
Most recipes use two layers, width 96, `ffnx=8`, AdamW learning rate `3e-4`, and weight decay
`0.01`. Study-adverse uses width 68; H&M user-churn uses `1.5e-4`; DNF uses `2.4e-3` and no
weight decay. Category vocabularies can grow during fitting, so parameter counts are not fixed.

Each task trains once on the complete inner training table. Full-split validation AUROC
(classification) or MAE (regression) selects the checkpoint and drives early stopping.
Training has finite update budgets and a 40-epoch cap; two full passes gate early stopping,
subject to those budgets. Validation runs at the recipe interval, capped at one epoch.
There is no train-plus-validation refit: the selected inner checkpoint predicts the masked
outer table using its officially supplied outer database context.

SQL applies per-anchor time cutoffs within each officially censored database, retains recent
histories in chronological order, and preserves the physical anchor row order. Feature queries
receive no labels. Static dimension values follow the supplied snapshot. For the two study
tasks, the known-empty `studies.limitations_and_caveats` column is restored as typed `NULL`
only when official cleaning removed it; supplied values are preserved. Other missing columns
fail explicitly, without consulting an alternate database.

The system observes RelArena's soft end-to-end runtime budget, reserving its final fifth for
checkpoint verification and outer preparation/prediction. Preparation, text encoding,
validation and prediction count toward the shared 24-hour ceiling on the largest tasks.

## Dependencies and validation

The data version is `relbench==2.1.2`. RelFlow 0.3.5 is pinned at source revision
`65fec85a1193c4e8a44c59e16ba05046e3c00d0b`. Text fields use frozen MiniLM weights from
`sentence-transformers/all-MiniLM-L6-v2`, pinned at
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. They are downloaded separately, not vendored.
The optional environment pins `huggingface-hub==1.16.1` and `transformers==5.9.0`.

Tests use synthetic data for configuration, temporal isolation, row ordering, checkpoint
selection and runtime integration. This single-SDK integration defines a new comparison;
it claims no historical benchmark numbers. Official results require the maintainer evaluation
under RelArena's hardware and sealed-test protocol. No data or trained checkpoints are bundled.
