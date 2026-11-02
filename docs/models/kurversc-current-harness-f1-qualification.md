# Frozen current-harness F1 qualification

Two fresh author runs completed with the submitted KurveRSC source
`f5bf98a03cdaa2b0798f80b44854f4629aba4eb0`, GraphReduce
`21aff2e9ad83d69d7fc468d01e3d56a0dc6ca976`, and RelArena Python code from
`0cec60d15e6132e1fc24b7edeffca3c15bd75133`. The current PR adds documentation
to that harness revision without changing the adapter, source pins or defaults.

The registered `KurveRSC20261022System` ran through
`run_system_experiment` with the caller's censored inner/outer splits, fresh
output directories, seed 0 and the unchanged 23.75-hour task budget. Predictions
were completed and hashed before a separate evaluation used the same
`EntityTask.evaluate(predictions, None, metrics)` call as the official runner.
Original request order and source fingerprints were verified. No model was
refitted for scoring, and no task-admission gate consumes these test scores.

| Task | Metric | Test score | Complete process hours | Peak process-tree RSS GiB | Minimum available system GiB |
|---|---|---:|---:|---:|---:|
| rel-f1/driver-top3 | ROC-AUC | 0.8341542120 | 2.709 | 11.442 | 17.906 |
| rel-f1/driver-position | MAE | 3.9778473960 | 3.135 | 10.025 | 18.965 |

Both runs completed without a memory-guard or deadline interruption and retained
the incumbent optional readout. The automatically selected graphs were depth 2
base + state for top3 and depth 1 base + conditional for position; both enabled
base-distribution features. These are recorded selections, not task dispatch.
The frozen source blueprints record explicit strict event boundaries for every
timed source instance; independent label-availability boundaries remain inclusive.

The machine was an arm64 Mac with 48 GiB RAM and an 8 GiB system reserve.
RSS is a sampled sum over the process tree, so shared pages may be counted
more than once; it is not cgroup memory or PSS. Queue wait is excluded from the
process times above. The numerical environment was already installed, using
Python 3.11.3; this does not establish a cold installation under the Python 3.12
lock. Exact package versions are in the JSON receipt.

Both scores are disclosed: position is worse than the preceding narrowed
pilot's 3.9031884144. Source/harness changes were not independently ablated, so
the score changes are not attributed to one correction. The earlier
[six-task pilot](kurversc-staged-results.md) remains tied to its original source.

[Machine-readable qualification](kurversc-current-harness-f1-qualification.json)
and [checksummed metadata evidence](kurversc-current-harness-f1-qualification-evidence.tar.gz)
include selections, source binding, scoring and resource receipts. Absolute
author paths are redacted; the archive index records original and published
digests. It contains no source data, prediction vectors or held-out targets.

No experimental research head was used. These two runs do not establish a fresh
21-task score, an Elo rank or independent maintainer reproduction. The
maintainer's harness rerun remains the reference result.
