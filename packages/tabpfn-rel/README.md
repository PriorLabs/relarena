# TabPFN-Rel

TabPFN-Rel predicts entity outcomes from relational tables. It builds deep feature
synthesis (DFS) features, adds optional calendar, history and text features, and
fits TabPFN using random or recency-weighted contexts.

This package contains the model and its search spaces. RelArena Core supplies the
relational data contracts, feature cache, tuning and predictive interface.

## Install

Requires Python 3.11 or 3.12. Choose a backend:

```bash
pip install "tabpfn-rel[local]"   # local TabPFN; GPU recommended
pip install "tabpfn-rel[api]"     # hosted TabPFN; requires API authentication
```

Base `tabpfn-rel` installs the model code and RelArena Core. The RelArena benchmark
package is optional. An inference extra supplies
the DFS engine and the selected estimator backend. Backend authentication and model
access follow TabPFN or tabpfn-client's own setup instructions.

Try the [generated database example](examples/tiny_database.py) from the monorepo root:

```bash
uv run --package tabpfn-rel --extra local python packages/tabpfn-rel/examples/tiny_database.py
```

It creates four customers and their event history, fits the default configuration,
and prints one prediction per customer.

## Predict on your database

Define the database relationships and prediction task in YAML, following
[RelArena's predictive interface](https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md).
Then:

```python
from tabpfn_rel import PredictiveQuery, PredictiveQuerySpec

spec = PredictiveQuerySpec.from_yaml("task.yaml", data_dir="data")
query = PredictiveQuery(spec).fit("tabpfn-rel-local", n_trials=0)
predictions = query.predict()
```

Use `tabpfn-rel-client` for the hosted backend. `n_trials=0` fits the default
configuration once; a positive budget enables the shared temporal tuning protocol.
API fits and predictions consume service quota. The predictive classes are the
same classes exported by `relarena_core.userdb` and, when installed,
`relarena.userdb`.

## Benchmark with RelArena

```bash
pip install "relarena[tabpfn-rel-local]"
relarena --model tabpfn-rel-local --datasets rel-f1 --tasks driver-dnf --n-trials 1
```

For this fixed-grid model, the CLI uses `--n-trials 1` for the default configuration.
Larger budgets also evaluate deeper DFS configurations. Unlike RPI, a zero CLI
budget evaluates no grid configurations.

`relarena[tabpfn-rel-api]` installs the hosted backend, selected with
`--model tabpfn-rel-client`.

The CLI and predictive interface discover installed models automatically. Direct
registry users call discovery explicitly:

```python
from relarena_core import discover_models, registry

discover_models()
model_class = registry.get("tabpfn-rel-local")
space = registry.search_space("tabpfn-rel-local")
```

For a custom training loop, import `TabPFNRelLocalModel`, `TabPFNRelClientModel`,
and their `TABPFN_REL_LOCAL_SPACE` / `TABPFN_REL_CLIENT_SPACE` directly from
`tabpfn_rel`. They implement RelArena's `fit` / `predict` model contract.
Importing `tabpfn_rel` or `relarena_core` does not discover plugins or load
estimator backends.

## Development

TabPFN-Rel is an independently installable distribution developed alongside
`relarena-core` and `relarena` in the RelArena monorepo. Its runtime dependencies
contain core and the selected backend; it does not require the benchmark package.
Each member has its own package metadata, source tree and tests. The repository
root owns the shared development lockfile, lint configuration and CI.

Run from the repository root:

```bash
uv sync --locked --all-packages
OMP_NUM_THREADS=1 uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files
uv build --package tabpfn-rel
```

The workspace source overrides resolve local members. Published wheel metadata
contains ordinary version requirements. The candidate distributions must be
released before the index-only installation commands above are available.

The tests exercise feature and context behavior without downloading model weights
or making hosted API requests. Integration tests use real DFS and a small test
estimator to cover the predictive interface and temporal tuning.
