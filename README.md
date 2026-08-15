# RelArena-α

A unified, fair benchmarking framework for running models on relational tasks on
[RelBench](https://github.com/snap-stanford/relbench) databases — inspired by how
[TabArena](https://tabarena.ai) standardizes tabular benchmarking.

This repository also open-sources TabPFN-Rel and an initial version of the
Relational Predictive Interface (RPI). A detailed release report covering
RelArena-α, TabPFN-Rel, and the RPI is in preparation.

> **Current status:** alpha release. RelArena is a living benchmark: its task coverage,
> baselines, API, and tuning regime will evolve with community feedback. The
> current release focuses on RelBench v1 entity-level forecasting tasks.

## Why

Reproducibility varies across relational-learning methods: some releases omit
training scripts or tuning details, and reported results often use different
evaluation and tuning regimes. RelArena provides one executable
*train → tune → evaluate* path with common data loading, split construction,
model selection, and result recording. Models provide their training code and a
declarative hyperparameter search space; callers choose the tuning budget.

## Core idea

```
┌─ runner ───── fit config(s) on train → pick best on val → final fit → test
├─ tuner ────── random search or a fixed grid under a caller-supplied budget;
│               records configurations, metrics, predictions, and phase timings
├─ model ────── RelArenaModel: fit / predict  (the contract)
├─ space ────── SearchSpace: what to tune over, bound to the model in the registry
└─ RelBench ─── Database, EntityTask, task.evaluate, metrics  (dependency)
```

**Evaluation protocol.** Each candidate configuration is fit on `train` and
scored on `val`. The best validation configuration then receives a final fit and
produces the test prediction. Depending on the model's published protocol, that
final fit either refits on `train + val` or trains on `train` while retaining
`val` for checkpoint selection. Parameter-free models simply run their sole
configuration. Test labels are withheld from the model and supplied only to
RelBench's evaluator.

RelArena uses **nested temporal validation**: tuning receives a database censored
at `val_timestamp`, while final evaluation receives one censored at
`test_timestamp`. This prevents access to post-boundary data and test labels.
Within that allowed database state, each method decides whether and how to enforce
the finer timestamp of every historical example. See
[docs/temporal-validation.md](docs/temporal-validation.md) for the complete
guarantee and trade-off.

Three design decisions carried over from TabArena/TabRepo:

1. **The search space is decoupled from the model and declarative.** Models
   implement just `fit` / `predict`; *what* to tune lives in a separate
   `SearchSpace` (a `ConfigSpace.ConfigurationSpace` for random search, or an
   explicit ordered `grid` for discrete spaces) bound to the model in the registry
   — mirroring AutoGluon / TabArena rather than declaring the space on the class.
2. **Budget is centralized rather than hidden in the model.** A model declares
   only its `SearchSpace`; the caller supplies `n_trials`. The alpha release uses
   documented, method-specific budgets because equalizing compute across methods
   remains an open problem. See [docs/tuning-regime.md](docs/tuning-regime.md).
3. **Runs retain useful metadata.** Each trial records its configuration, metrics,
   optional predictions, and separate tuning/final-fit timings for later analysis.

## Layout

```
src/relarena/
  model.py        # RelArenaModel — the contract every model implements
  search_space.py # SearchSpace — declarative HPO space (ConfigSpace or grid)
  registry.py     # string-keyed model registry, binds model -> search space
  tasks.py        # entity task-type scope + guard
  metrics.py      # metric direction map + primary-metric selection
  tuner.py        # random search / fixed grids; per-trial timing and predictions
  runner.py       # local orchestration for one (model, dataset, task)
  results.py      # TrialResult schema + DataFrame export
  models/         # constant, lightgbm, rdblearn, graphsage, relgnn, relgt, tabpfn-rel, rt wrappers
  featurization/  # relational DB -> flat feature table (entity-only, for now)
  checksums/      # content fingerprints of the RelBench data + the recorded baseline
  evaluation/     # leaderboard, plots, externally-reported reference baselines
  userdb/         # Relational Predictive Interface (RPI)
tests/            # smoke + unit tests (no data download)
```

## Models

This is the canonical inventory of registered methods. The release snapshot and
paper contain the rows marked **paper**; the additional `relgnn` registration
is retained as an experimental final-fit variant.

### Models and systems

Every method registers as one of two kinds (`RelArenaModel.kind`), and the two
are **not the same kind of result**. Both face the same tasks, splits, metrics,
and runtime budget, so the comparison is fair on final performance: if a system
scores higher, it really did do better on the benchmark. What a system gives up
is the controlled setup. A **model** is one method under the harness's fixed
tuning pipeline, so its score isolates the method. A **system** is free to step
outside those constraints — it selects its own hyperparameters, training
schedule, or components inside `fit` — so its score tells you what the whole
package achieves without telling you which part earned it: how much comes from
the underlying architecture rather than the selection machinery or other
transferable tricks is not identifiable from the benchmark alone. Leaderboards
should either exclude systems (`compute_leaderboard(..., kinds={"model"})`) or
rank both populations together with systems clearly marked; publishing both
boards side by side is the recommended presentation.

System support is currently **highly experimental**: systems run through the ordinary
model API with documented workarounds (all tuning inside a single fit of the
default config, state carried between the fit and refit phases via module-level
globals), and a system submission needs extra validation by and discussion with the maintainers. A
future release will replace these workarounds with an explicit fitting API for
systems — see
[adding-a-model.md](docs/adding-a-model.md#model-or-system) for the current
rules.

| Registered identifier | Paper-facing name | Family | Kind | Status | Final fit | Extra |
| --- | --- | --- | --- | --- | --- | --- |
| `constant-global` | Constant (global) | global constant | model | paper | train + val | core |
| `constant-per-entity` | Constant (per-entity) | entity-wise constant | model | paper | train + val | core |
| `lightgbm` | LightGBM | entity-only tabular | model | paper | train + val | `lightgbm` |
| `rdblearn` | RDBLearn | DFS + tabular foundation model | model | paper | train; val retained | `rdblearn` |
| `tabpfn-rel-local` | TabPFN-Rel (OSS) | DFS + TabPFN v3 | model | paper | train + val | `tabpfn-rel-local` |
| `tabpfn-rel-client` | TabPFN-Rel (API) | DFS + hosted TabPFN v3 with text | model | paper | train + val | `tabpfn-rel-api` |
| `graphsage` | GraphSAGE | relational GNN | model | paper | train + val | `graphsage` |
| `relgnn-es` | RelGNN | relational GNN | model | paper | best-validation checkpoint | `relgnn` |
| `relgnn` | RelGNN full-data refit | relational GNN | model | experimental variant | train + val | `relgnn` |
| `relgt` | RelGT | relational transformer | model | paper | best-validation checkpoint | `relgt` |
| `rt-plurel` | RT-PluRel | pretrained relational transformer, fine-tuned per task | **system** | paper | train + val | `rt` |

The paper reports `relgnn-es` simply as **RelGNN**, because that published-style
best-validation-checkpoint regime performed better in our runs. The regular
`relgnn` identifier remains available for experiments but is excluded from the
default release leaderboard.

RT-PluRel is the sole registered **system** (see
[Models and systems](#models-and-systems)); its protocol and every configured
value are documented in [`models/rt/model.py`](src/relarena/models/rt/model.py).
Note that its recorded `val_score` is a placeholder — see
[`baseline_results/README.md`](baseline_results/README.md).

Everything else per method lives at its source: install caveats in
[docs/adding-a-model.md §6](docs/adding-a-model.md#6-optional-dependencies)
(the GNN baselines need platform-specific PyG sampling wheels beyond their
extras), excluded backends and cache warmers in each model's docstring, and
the complete implementation choices in the
[adding-a-model appendix](docs/adding-a-model.md#appendix--what-every-existing-model-chose).

## Install & test

```bash
uv sync            # the dev group (pytest, ruff, ...) installs by default
OMP_NUM_THREADS=1 uv run pytest  # the prefix is required on macOS; harmless elsewhere
```

The `rt` extra currently installs a platform-specific wheel directly from a
GitHub release. Public package indexes reject distribution metadata containing
direct-URL dependencies. Before publishing RelArena to PyPI, publish
`relational-transformer` there as a normal versioned dependency or remove the
direct reference from RelArena's published metadata.

## Use RelArena on your own database (RPI)

The **Relational Predictive Interface (RPI)** applies registered RelArena models
to an entity-level forecasting task over your own relational database. Describe
CSV or Parquet tables in a YAML database specification, define the forward-looking
label and split boundaries in a YAML task specification, then use
`PredictiveQuery` as the Python façade:

```python
from relarena.userdb import PredictiveQuery, PredictiveQuerySpec

spec = PredictiveQuerySpec.from_yaml("task.yaml", data_dir="data/")
predictions = PredictiveQuery(spec).fit("tabpfn-rel-client").predict()
```

See [docs/predictive-task.md](docs/predictive-task.md) for the task definition,
SQL rules, split semantics, and worked examples.

## Preprocessing caches (optional)

Expensive CPU-bound preprocessing may run before a benchmark and be reused across
trials. This can substantially reduce repeated-run time, especially for DFS feature
matrices, materialized graphs, and tokenized databases.

Caching is not required. RelArena provides an **optional, experimental** helper API
in [`relarena.cache`](src/relarena/cache.py) for local paths, miss policies, private
scratch computation, and atomic publication. A method may ignore this API and
implement caching independently. The helper does not bring cache warming into a
timed RelArena experiment; preprocessing scripts still run separately, so their
runtime is not currently included in the recorded experiment timings.

Regardless of the mechanism, cache-generation code must be public, reproducible,
and leakage-safe. Some practical pointers:

- Load data through `RelBenchDatasetTask.inner_split()` and `outer_split()` so the
  validation and test phase boundaries remain intact.
- Let the preprocessing implementation own its keys, versions, serialization, and
  validation; include only inputs that actually determine the artifact.
- Treat pre-built stores as a convenience: always ship a runnable warmer that can
  reconstruct them.

The full implementation guidance and reference code live in
[docs/adding-a-model.md](docs/adding-a-model.md#4-pre-processing-cache).

Configure the store explicitly at the run entrypoint:

```python
run_experiment(..., cache_dir="~/relarena-cache")
```

Entrypoints also resolve these environment variables once:

- `RELARENA_CACHE_DIR` — the store directory.
- `RELARENA_DISABLE_CACHE` — set to any value to disable persistent caches.

`RELARENA_DISABLE_FEATURE_CACHE` remains as a deprecated alias for one release.
The helper API's store is an ordinary local directory; remote snapshot transport
belongs to deployment infrastructure rather than RelArena itself.

**Precompute (CPU).** Because `fit` / `predict` read (a miss raises), build the store
up front with a fill run. The DFS engine (`fastdfs`) runs on CPU and is memory-hungry
on wide-fan-out schemas, so run it on a large CPU node (many cores, ample RAM);
everything after it runs on the GPU (or the hosted TabPFN API), so precomputing keeps
that CPU-heavy step off those nodes. The workflow warms every RelBench v1 task:

```bash
RELARENA_CACHE_DIR=~/relarena-cache \
    uv run --extra rdblearn python workflows/warm_feature_cache.py
```

It invokes `relarena.featurization.warm_cache` for both protocol splits and warms
both legitimate outer histories: train-only for RDBLearn and train+val for models
that refit on all labeled data. The `tabpfn-rel` and `rdblearn`
models share full-anchor, leak-safe-history matrices whenever their actual inputs
match; model-specific row selection and downstream training do not affect the key.
On a warm cache the evaluation reads Parquet only — no RDB build and no DFS.
RelGNN, RelGT, and RT-PluRel expose independent runnable warmers at
`relarena.models.relgnn.warm_cache`, `relarena.models.relgt.warm_cache`, and
`relarena.models.rt.warm_cache`.

**Runnable demo.** `examples/tabpfn_rel_caching.py` fits one RelBench task with and
without a precomputed cache, reports both timings, and checks the outputs are
identical. Its header includes a CPU-only mode (`RELARENA_EXAMPLE_SKIP_TFM=1`)
that exercises the DFS and cache path without a GPU.

## Batch evaluation

The CLI runs one model across many tasks in-process and writes every evaluated
config to a CSV:

```bash
relarena --model lightgbm --datasets rel-f1 --output results.csv
```

Each `(model, dataset, task, seed)` experiment is independent, so sweeps
parallelize trivially. The building blocks are all public — `run_experiment`
executes one experiment, `summary_to_dataframe` flattens it into the shared
results schema, and concatenated frames feed the leaderboard (needs the
`leaderboard` extra):

```python
import pandas as pd

import relarena.models  # registers the built-in models
from relarena.evaluation import compute_leaderboard
from relarena.registry import registry
from relarena.results import summary_to_dataframe
from relarena.runner import run_experiment
from relarena.tasks import list_entity_tasks

frames = []
for spec in list_entity_tasks(["rel-f1"]):
    for model in ("constant-global", "lightgbm"):
        summary = run_experiment(
            registry.get(model), spec.dataset, spec.task, seed=0, n_trials=10
        )
        frames.append(summary_to_dataframe(summary))
board = compute_leaderboard(pd.concat(frames, ignore_index=True))
```

If you have a large-scale cluster, integrate that loop into your distributed
backend of choice (a SLURM array, Ray, ...): dispatch each experiment as one
job, cache each job's result frame keyed by `(model, dataset, task, seed,
n_trials)`, and concatenate the cached frames for the leaderboard. Warm the
shared caches first (`workflows/warm_feature_cache.py` and the per-model
`warm_cache` modules) so workers never pay the featurization cost.

## Baseline results

[`baseline_results/`](baseline_results/) holds the release snapshot of the sweep
over the RelBench-v1 entity tasks: `results.csv` (every evaluated config; feed
it to `compute_leaderboard`) and `reference_results.csv` — per-task scores for
methods **not reproduced in this pipeline**, transcribed from published model
reports and flagged with a `_MR` (**model report**) suffix. `_MR` numbers are
mostly self-reported and are often higher than the results reproduced through
RelArena; we discuss possible reasons in the forthcoming release report
of this README. With few exceptions, these results are
not directly comparable to RelArena runs and should only be used as reference
points. To include them in a leaderboard or plot, pass `reference=`
(`relarena.evaluation.load_reference_results`). See
[`baseline_results/README.md`](baseline_results/README.md) for per-method
provenance and caveats.

## Adding a model

A model is a folder under `src/relarena/models/` implementing the
`RelArenaModel` contract (`fit` / `predict`) with a `SearchSpace` registered via
`@register_model(search_space=...)`; the registry discovers the folder
automatically. `models/lightgbm/` is the smallest complete example to copy.

The full guide is [docs/adding-a-model.md](docs/adding-a-model.md): the layout,
the datatypes `fit` and `predict` receive, the tuning regime and its choices,
where shared code goes, optional dependencies, vendoring requirements, tests,
and a checklist.

## License

Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Two things the license on
this code does not settle, both worth reading before you rely on relarena:

- **`tabpfn` is not Apache-2.0.** It ships the Prior Labs License, an Apache-2.0
  derivative whose added paragraph 10 requires anyone distributing a product built
  on it to display "Built with PriorLabs-TabPFN". It is confined to the `rdblearn`
  and `tabpfn-rel-*` extras, so a plain install does not pull it.
- **Datasets are not ours to license.** relarena serves no data itself; `relbench`
  downloads every database at runtime, and they remain subject to their own
  upstream terms.

See [`docs/licensing.md`](docs/licensing.md) for what the license does and does not
cover, and [`NOTICE`](NOTICE) for third-party attribution.
