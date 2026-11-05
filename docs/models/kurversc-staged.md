# KurveRSC: generic staged system

`kurversc-2026-10-22` is the versioned system; `kurversc-latest` resolves to it.
The earlier native `kurversc` registration remains distinct. The thin adapter
passes caller-owned censored inner/outer splits to
`kurversc.protocol_runner.run_protocol_task`; RelArena owns hidden-label scoring.

## Procedure

1. Schema-derived graph search with bounded 10K/50K screening and full-frame
   finalist evaluation, including CDF/base representations and peer context.
2. A bounded production-faithful finalist comparison for **both binary
   classification and regression**, followed by frozen operation-plan replay.
   The same 1% relative-loss cap uses `1 - AUC` or MAE for candidate screening
   and every seed/fold. At most one graph with at most half the incumbent's
   columns is compared. Replacement requires unanimous native-selector
   decisions across seeds 0/1/2 on two or three temporal origins.
3. Historical temporal-training-policy selection on up to three origins and
   seeds 0/1/2; deployment seed 0.
4. One optional menu: generic parent text, three/five-frame ensembles,
   target-domain projection, zero-aware readout and object breadth. Applicable
   candidates use the same incumbent and `select_temporal_policy` gate;
   eligibility is independent of previous admissions.
5. Validation-locked fitting and keyed prediction.

The source graph supplies the repeated-table bounds offered by its candidate
menu. Frozen replay restores and verifies source limits, aliases, keys, clocks
and relationships. Source event timestamps are strictly before each query;
availability and label-maturity clocks retain their declared semantics.
Masked forecast requests keep their original keys and order, including requests
whose root payload is unavailable at a strict cutoff.

Exposure-role reconstruction, recurrence/semantic corrections, conditional-count
recovery, production-sequence recovery, bounded views and test-win campaign gates
are excluded from the packaged procedure.

## Installation and execution

The workspace installs exact PyPI releases: `kurversc[staged]==0.2.0` and
`graphreduce==1.10.18`. Neither package uses a Git dependency override.

```bash
pip install "kurversc[staged]==0.2.0" "graphreduce==1.10.18"
```

```bash
uv sync --all-packages --group cpu --extra kurversc --python 3.12
KURVERSC_DUCKDB_MEMORY_LIMIT=6GiB KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE=128GiB OMP_NUM_THREADS=1 uv run --no-sync relarena --model kurversc-latest --datasets rel-f1 --tasks driver-dnf --seed 0 --output kurversc_driver_dnf.csv
```

The entire procedure shares a deadline below 24 hours, retained across resumes.
Incomplete stages raise rather than returning partial/native predictions.
Independent tasks can run concurrently with separate output, scratch and cache
directories and RAM headroom; preserve the internal worker/thread settings.
See [resource planning](kurversc-staged-resources.md).

## Reproduction and provenance

The maintainer's 21-task reproduction through this registered adapter is the
reference result. It ran at seed 0 with kurversc 0.2.0 and graphreduce 1.10.18
from PyPI, one task per CPU host, and its rows are the `kurversc-2026-10-22`
rows in [`baseline_results/results.csv`](../../baseline_results/results.csv).
Historical author results and large evidence bundles are kept outside this
submission's working tree; they remain available in the prior PR history and
the KurveRSC repository.

The [component definitions and global defaults](https://github.com/kurveai/kurversc/blob/cf6599bd7de21f32b299c3cdce75f611ad050a06/docs/generic-staged-review.md)
and [source/package qualification](https://github.com/kurveai/kurversc/blob/cf6599bd7de21f32b299c3cdce75f611ad050a06/docs/generic-staged-validation.json)
are documented in [KurveRSC #5](https://github.com/kurveai/kurversc/pull/5).
Qualification: 727 source tests passed with one skip; 19 adapter/discovery checks
passed; installed-wheel classification/regression runs each completed the whole
registered procedure on a fresh 48-request schema. The submitted pin keeps
ongoing experimental feature work outside this reproduction.
