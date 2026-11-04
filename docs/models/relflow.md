# Relflow

## Build and adapt foundation models for your own structured data.

https://www.relflow.ai/en/latest/

https://github.com/relflow/relflow

Relflow is an open source model factory for typed, nested data.
Describe your records and prediction targets, and it generates
a custom neural architecture, trains the model, and
serves predictions and learned representations.

`relflow` fits one frozen shared recipe per size tier. The harness controls
validation, the train+validation refit and hidden-label scoring. There is no
per-task architecture search, deduplication change, mixed cropping or unreduced
context. The optional backend requires Python 3.12 and `relflow[text]==0.4.4`.

Shared size-tier presets kept research compute to approximately **US$1,500** in Modal compute
while producing competitive research results across the task
suite. Presets are frozen for submission.

## Fixed modeling policy

Tier depends only on the original inner training-anchor count and is carried
through the final refit: SMALL below 20,000, MEDIUM below 500,000, LARGE otherwise.
Fields, joins and datatypes are derived from the supplied relational structure
and validation-censored column measurements. Raw values and timestamp ages are
retained; no manually engineered aggregate features are supplied.

| Setting | SMALL | MEDIUM | LARGE |
| --- | ---: | ---: | ---: |
| Width / root layers / heads | 128 / 2 / 4 | 64 / 2 / 8 | 64 / 2 / 8 |
| Root feed-forward multiplier | 2 | 4 | 4 |
| Branch layers / heads / multiplier | 1 / 4 / 8 | 2 / 4 / 8 | 2 / 4 / 8 |
| Branch summary vectors | 4 | 1 | 1 |
| Decoder layers / heads / multiplier | 1 / 4 / 8 | 1 / 4 / 8 | 1 / 4 / 8 |
| Learning rate | 0.0001504508367492947 | 0.00021681002964478338 | 0.00021681002964478338 |
| Weight decay | 0.01 | 0 | 0 |
| EMA | 0.995 | 0.995 | 0.995 |
| Leaf / branch masking | 0.30 / 0.10 | 0.30 / 0.05 | 0.30 / 0.05 |
| Additive jitter | 0.05 | 0 | 0 |
| Model outer / inner caps | 128 / 32 | 64 / 32 | 32 / 16 |
| SQL outer / inner pools | 512 / 64 | 256 / 64 | 128 / 32 |

All nodes have zero neural dropout. AdamW uses a constant learning rate,
gradient clipping 0.25 and EMA inference weights. Training runs eight complete
pass-equivalents; MAE is the regression objective. `overflow="random"` samples
contiguous training windows and keeps deterministic tails during validation and
prediction. Attachment multiplicity remains unchanged.

## Operational policy

These settings are fixed resource choices, not model-selection candidates.
They preserve the recorded effective batch sizes from the singleton campaign:

| Tier / task | Training batch | Loader workers | Training shuffle rows |
| --- | ---: | ---: | ---: |
| All SMALL | 64 | 0 | 16,384 |
| MEDIUM user-clicks, user-visits, site-success | 256 | 0 | 16,384 |
| MEDIUM study-adverse; unseen MEDIUM tasks | 512 | 0 | 16,384 |
| All LARGE | 512 | 2 | 2,048 |

LARGE CUDA loaders use persistent spawned workers, one prefetched batch per
worker and no pinning. SMALL and MEDIUM use the main process. CPU execution
always uses zero workers. Every source pass reopens Parquet as 8,192-row
record batches without Arrow threading, preserving the archived source/shuffle
geometry; the restartable factory is safe to send to spawned workers. Text
token caches are bounded to 16,384 strings per field/process on LARGE and 4,096
on other tiers, with relflow's 64 MiB payload bound. Frozen encoder vectors use
the extension-owned disk store with explicit resident-vector budgets of 256 MiB
for SMALL/MEDIUM and 2,048 MiB for LARGE, per store view. The task-name exceptions
affect only batch size, not schema, neural architecture or optimizer settings. An OOM
fails visibly; it does not silently lower a frozen batch, truncate history or substitute an incomplete training run.

Text uses `sentence-transformers/all-MiniLM-L6-v2` at immutable revision
[`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41).
Its model/tokenizer snapshot is resolved only when a Text field is present,
inside the fit budget; every field with that encoder reuses the same snapshot.

DuckDB uses eight threads, with memory caps of 4/16/32 GB and spill caps of
64/128/256 GB for SMALL/MEDIUM/LARGE. These are maximum allowances, not an
allocation guarantee; available host memory and scratch space require checking.

The extra is separate from KurveRSC 0.1.1 because relflow requires NumPy 2 while
that version requires NumPy 1. The root lock declares conflicts with both the
KurveRSC extra and its development dependency group. Use an isolated environment
with the `relflow` extra; Python 3.11 and base discovery remain supported without
the backend.

## Input, output and timing boundaries

The adapter only consumes the `Database` and `Table` objects supplied by the
harness. It never reloads a raw RelBench cache or calls dataset/task getters.
Schema profiling uses the validation boundary, including during the outer fit.
Training anchors use allowed outer history, while validation anchors in the
final union retain history frozen at the validation boundary, matching the
original campaign. Per-anchor timestamps further censor histories and lookups.

Prediction returns one finite scalar or positive-class probability in each
request's original position; empty requests return an empty vector. Missing
values and empty histories retain native Arrow structure. Decimal numeric
inputs are projected to double precision for the neural model; Decimal category
labels are projected to exact strings, preserving distinct identities. Integer
identifiers and categorical integer values retain their original representation.
The adapter does not load research checkpoints, observe test labels or choose
between model scores.

Preparation, text encoding and training occur inside `fit`; prediction is a
separate harness call. Their full elapsed time must remain in the run record.
The official benchmark runner still performs an inner diagnostic fit and an
outer refit for a singleton; published default-only timing excludes that inner
fit. Both actual elapsed work and published timing must be retained until the
maintainer confirms how the 24-hour policy applies. CPU memory, scratch capacity,
GPU kernels, text-store behavior and source/schema equivalence still require
qualification on the actual reproduction machine. No score or bitwise equality
with historical campaign runs is asserted.

Cold relational preparation and text encoding remain inside the timed calls.
A public cache-warming workflow and its accounting require maintainer policy
clarification; this implementation does not silently move preparation outside
the measured run. Maintainers own `baseline_results` updates. See the
[integration guide](../adding-a-model.md),
[temporal protocol](../temporal-validation.md) and
[runtime policy](../tuning-regime.md).
