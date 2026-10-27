# KurveRSC complete staged candidate

`kurversc-2026-10-22` is a versioned system using the complete packaged staged
procedure; `kurversc-latest` resolves to it. The existing `kurversc` native
graph-only registration remains distinct and its historical result identity
is not relabeled by this submission.

```bash
uv sync --all-packages --group cpu --extra kurversc --python 3.11.17
KURVERSC_DUCKDB_MEMORY_LIMIT=123GiB OMP_NUM_THREADS=1 uv run --no-sync relarena \
  --model kurversc-latest --datasets rel-f1 --tasks driver-top3 \
  --seed 0 --output kurversc_staged_driver_top3.csv
```

Omit the dataset/task filters to execute the full supported panel. The system
owns its internal search, so model tuning trial counts do not control it.
The candidate currently requires the pinned KurveRSC/GraphReduce release
branches until their versions are published.
The workspace pins both source commits in `tool.uv.sources`; it does not
resolve an older PyPI package sharing a nominal experimental version.

## Procedure and boundaries

The adapter calls the public `kurversc.protocol_runner.run_protocol_task`
with the actual censored inner/outer splits. It does not download a separate
database, inspect published winners, score test labels, or substitute a
native-only graph run when the complete procedure fails. Predictions retain
the original masked request order; the RelArena scorer owns test evaluation.

The packaged procedure uses bounded 10K/50K screening and full-frame finalist
evidence, freezes the selected operation plan, and evaluates optional temporal
support/readouts using available retrospective origins and fixed seeds 0/1/2.
Deployment seed is 0; unsupported seeds raise explicitly. Preprocessing,
search, selection, refitting and prediction share one original budget below
24 hours; checkpoint resume does not reset that clock.

`kurversc.run_staged_task` also accepts explicitly constructed task contexts
on unregistered schemas. Ordinary `kurversc.fit`/`predict` remain lower-level
library interfaces, not aliases for the entire staged procedure.

## Review qualification and resource planning

The fresh frozen installed-wheel campaign completed **all 21 tasks**, with no
failures or historical score substitutions. The [final score comparison](kurversc-staged-results.md)
and [resource report](kurversc-staged-resources.md) contain the complete measurements.
The official scorer gives 1948.6 point Elo versus 1937.1 for the captured
September 28 client, with overlapping uncertainty intervals. This replaces the
old KurveRSC entry on that fixed panel; it does not silently add an extra method.
`baseline_results/` remains unchanged for independent maintainer reproduction.

The earlier cache/rematerialization and SQL memory failures have been repaired
in the pinned companion implementations. Narrow projections, disk-backed
historical capture, bounded normalization and equivalent cache comparisons
preserve the configured evidence. Scoped wide-aggregate thread control avoids
multiplying independent distinct-aggregation state and restores the caller's
thread setting. Source hashes match the frozen score-producing package payload
before release-version stamping; old cache identities are not forged.

See the [per-task resource report](kurversc-staged-resources.md) and its CSV for
full-task elapsed time, kernel cgroup RAM peaks, sampled tree RSS/CPU/storage
and measurement coverage. The host provides about 496 GiB RAM and 64 logical
CPUs. The largest observed task is user-badge at 333.51 GiB and 14.17 hours;
these are observations, not a minimum-hardware certification. The campaign
uses eight learner threads per task, one graph-search worker and one seed
worker. Larger hosts can run independent tasks concurrently using the report's
memory/CPU/storage admission rules. No arbitrary 64-GiB total-task kill cap is
part of the measured campaign.

The staged extra pins KurveRSC 0.2.0 and GraphReduce 1.10.18 plus the qualified
numerical/learner versions. Both source commits are locked for review before
PyPI publication; an experimental version number alone does not identify the
qualified code. The qualified Linux environment uses Python 3.11.17.

The optional text encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with safetensors and no remote
model code. Its external pretraining includes StackExchange data; eligibility
remains subject to the benchmark's external-data review. Retaining the original
incumbent is a selection safeguard, not a guarantee against all unseen holdout
regressions.

Package validation passes 704 KurveRSC tests, 157 GraphReduce tests (2 skipped)
and 19 current RelArena adapter/discovery tests. Full installed-wheel
classification/regression smokes through the current registered class and actual
core split types reproduce the frozen wheels' keyed predictions exactly on an
unregistered 48-request schema. They use the existing qualified dependencies;
they do not establish a cold resolution of every optional package. The full
current-harness 21-task reproduction remains a maintainer
review/reproduction step; prior campaign scores are not silently relabeled as
new-harness scores. No main-branch merge or PyPI publication is performed by
this PR update.
