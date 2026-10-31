# KurveRSC: generic staged revision

`kurversc-2026-10-22` is a versioned system; `kurversc-latest` resolves to it.
The earlier native `kurversc` registration and historical result identity remain
distinct. Following maintainer review, the staged source pin now selects the
narrowed `kurversc-generic-staged-v1` recipe.

## Retained procedure

1. Schema-derived graph search with bounded 10K/50K screening and full-frame
   finalist evaluation.
2. Production-faithful finalist comparison and frozen operation-plan replay.
3. Historical temporal-training-policy selection using up to three origins,
   validation seeds 0/1/2 and deployment seed 0.
4. One optional menu: generic parent text, three/five-frame ensembles,
   target-domain projection, zero-aware readout and object breadth. Applicable
   candidates compare against the same temporal-policy incumbent through one
   `select_temporal_policy` gate. Earlier admissions do not determine eligibility.
5. Validation-locked fitting and keyed prediction.

Exposure-role reconstruction, recurrence/semantic teacher/ranker correction,
conditional-count recovery, production-sequence recovery, bounded views and
benchmark-test-win campaign gates are absent from the packaged execution closure.
That closure contains 32 stage modules, reduced from 108. The KurveRSC staged
extra no longer requests the removed pretrained semantic encoder dependencies.

[Component requirements and global defaults](https://github.com/kurveai/kurversc/blob/b2cbeea0c7ee26d22e7076c7424c7ab108b2d060/docs/generic-staged-review.md)
are documented with the companion revision in [KurveRSC #4](https://github.com/kurveai/kurversc/pull/4).
The retained presets are bounded global defaults, not claims that historical
validation erases prior benchmark-driven development.

## Inputs and scoring

The thin adapter forwards the caller's actual censored inner/outer splits to
`kurversc.protocol_runner.run_protocol_task`. It does not download another
database or obtain hidden targets. RelArena owns holdout scoring.

RelBench timed-table adaptation explicitly uses source event time strictly
before each query. Independent availability clocks remain inclusive at their
boundary. Strict filtering precedes annotation/materialization, and missing
root payloads do not remove forecast requests. Label support requires horizon
maturity and its declared availability. Saved predictions retain the original
masked request keys and order.

Preprocessing, search, selection, refitting and prediction share an original
budget below 24 hours. Resume retains that deadline. Deployment seeds other than
0 raise; incomplete stages raise rather than returning a native-only fallback.

## Installation and execution

The workspace pins KurveRSC 0.2.0 to the revised source commit and retains
GraphReduce 1.10.18 at its reviewed commit. A nominal version number alone does
not identify this revision. No PyPI publication is performed by this PR update.

```bash
uv sync --all-packages --group cpu --extra kurversc --python 3.12
KURVERSC_DUCKDB_MEMORY_LIMIT=6GiB KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE=128GiB OMP_NUM_THREADS=1 uv run --no-sync relarena --model kurversc-latest --datasets rel-f1 --tasks driver-dnf --seed 0 --output kurversc_driver_dnf.csv
```

The engine limits above match the local pilot. They govern spilling, not total
process memory. Use the [resource report](kurversc-staged-resources.md) to plan
admission. Independent tasks may execute concurrently with separate output,
scratch and cache locations; keep internal worker/thread settings fixed.

`kurversc.run_staged_task` also accepts an explicit `TaskContext` for a schema
outside RelBench. Native `fit`/`predict` remain lower-level component interfaces.

## Revised validation status

- Clean companion source: **710 passed, 1 skipped**.
- Current RelArena adapter/discovery checks: **19 passed**.
- Complete installed-wheel classification and regression smokes through this
  registered class and actual current-core split types: **48 requests each** on
  fresh unregistered schemas; no hidden targets supplied.
- Wheel/sdist contents, exclusions and predictive-source parity checks pass.
- The [six-task local pilot](kurversc-staged-results.md) has **six completed**
  official RelBench-scored results. Scores, admissions and sampled resources
  are disclosed, including decreases. Initial metadata/resume repairs are recorded.

These checks use existing numerical dependencies. They do not establish cold
installation or an independent current-harness 21-task reproduction. No revised
21-task Elo is claimed; `baseline_results/` is unchanged. The
[October 27 results](archive/kurversc-staged-20261027/kurversc-staged-results.md)
and resources are explicitly archived as measurements of the superseded recipe.

[Checksummed pilot admission evidence](https://github.com/kurveai/kurversc/blob/b2cbeea0c7ee26d22e7076c7424c7ab108b2d060/docs/generic-staged-pilot-admissions.tar.gz)
contains the saved selections, fold/seed reports and source receipts for the
completed tasks. It excludes source data, predictions and held-out targets.
