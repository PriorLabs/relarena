# KurveRSC complete staged candidate

`kurversc-2026-10-22` is a versioned system using the complete packaged staged
procedure; `kurversc-latest` resolves to it. The existing `kurversc` native
graph-only registration remains distinct and its historical result identity
is not relabeled by this submission.

```bash
uv sync --all-packages --group cpu --extra kurversc
OMP_NUM_THREADS=1 uv run --no-sync relarena \
  --model kurversc-latest --datasets rel-f1 --tasks driver-top3 \
  --seed 0 --output kurversc_staged_driver_top3.csv
```

Omit the dataset/task filters to execute the full supported panel. The system
owns its internal search, so model tuning trial counts do not control it.
The candidate currently requires the pinned KurveRSC/GraphReduce release
branches until their release-candidate versions are published.
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

## Draft qualification status

The most recent frozen campaign completed 12 of 21 tasks. This PR does not
claim a complete Elo ranking, does not substitute historical scores for missing
tasks, and leaves `baseline_results/` untouched for maintainer reproduction.

Before merge, resolve and independently retest:

- exact rematerialized float/category frame checks;
- a hidden host-quarter DuckDB working-memory cap, which caused failures even
  with host RAM available;
- the widest graph candidates' memory requirements, including a roughly
  414-GiB H&M churn peak on a roughly 496-GiB host;
- all 21 fresh completions, complete runtime accounting and peak process-tree
  memory, and actual full-pipeline reproduction under the current core layout.

The optional text representation uses a pretrained semantic encoder. Its model
revision/assets and dependencies must be pinned for reproduction and reviewed
under the benchmark's external-data policy. Retaining an incumbent candidate
does not guarantee zero future holdout regression.

No source rows, graph capabilities, learner budgets or full-fidelity evidence
were removed merely to make the unfinished tasks fit the machine.
