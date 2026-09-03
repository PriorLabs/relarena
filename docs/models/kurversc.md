# KurveRSC

KurveRSC is a CPU-only native RelArena system that jointly selects a GraphReduce relational
feature program and a CatBoost learner. It does not require or use a GPU.

## Installation

Sync RelArena's CPU dependency group and the KurveRSC extra:

```bash
uv sync --group cpu --extra kurversc
```

## Running KurveRSC

Run one task through the ordinary RelArena CLI:

```bash
OMP_NUM_THREADS=1 uv run --group cpu --extra kurversc relarena \
    --model kurversc \
    --datasets rel-stack \
    --tasks user-badge \
    --output kurversc_user_badge.csv
```

Run all 21 RelBench v1 entity classification and regression tasks by omitting `--datasets` and
`--tasks`:

```bash
OMP_NUM_THREADS=1 uv run --group cpu --extra kurversc relarena \
    --model kurversc \
    --output kurversc_all_tasks.csv
```

KurveRSC owns its GraphReduce search inside `RelArenaSystem.run`; the model-only `--n-trials`
option does not alter its internal search.

## RelArena configuration

Each phase receives RelArena's officially censored database. KurveRSC searches connected
point-in-time frames, freezes the selected GraphReduce operations, refits from the full phase
tables, and replays that plan for test prediction.

The submitted configuration:

- evaluates every admitted graph configuration on the complete latest-cutoff relational frame;
- reranks the top three candidates over three complete cutoff frames, processed sequentially;
- uses the latest eligible production frame for graph search and final learner fitting;
- limits each automatic feature family to four source columns per node; and
- applies an 8,000-feature pre-materialization width guard.

The bounded search explores GraphReduce feature-family combinations, graph depth, and automatic
annotation. It prunes candidates that exceed the width guard or cannot produce features for the
task schema. The fixed values live in
[`src/relarena/models/kurversc/model.py`](../../src/relarena/models/kurversc/model.py), so the
registered system name denotes one reproducible procedure without hidden configuration fields.
Use KurveRSC's public API for ablations or alternative frame budgets.

![KurveRSC RelArena default: complete latest-cutoff graph search, top-three reranking over three sequential complete cutoff folds, a frozen graph plan, and final fitting on one complete cutoff.](../kurversc-relarena-default.svg)

## Hardware

For the default sequential sweep, use a Linux CPU host with 96–128 GiB RAM and at least 150 GiB
of free fast local SSD or NVMe scratch space. A 64 GiB host may work by spilling to disk but is
closer to the limit on the largest tasks. Cluster schedulers can execute independent RelArena
tasks on separate CPU nodes when greater throughput is required.
