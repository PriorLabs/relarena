# Adding a model

Every decision point in adding a model, the values each can take, and which
existing model chose what.

A model is a folder under `src/relarena/models/`. Drop it in and the registry
auto-discovers it; there is no shared file to edit. The harness owns the loop
invocation and evaluation protocol, while the model supplies training code and a
declarative description of what to tune. `relarena` applies a common tuning
procedure and uses method-specific trial budgets and runtime constraints to
approximately align tuning compute across methods. Exact compute equivalence
remains difficult because methods differ substantially in per-trial runtime and
pre-processing cost. See [The tuning regime](tuning-regime.md) for the current
budgets, runtime policy, and remaining limitations.

## 1. Layout

| Variant | Files |
| --- | --- |
| Minimum | `mymodel/{__init__,model}.py` + `tests/mymodel/{__init__,test_model}.py` |
| Split out helpers | add e.g. `mymodel/features.py`, `mymodel/context.py` (`tabpfn_rel` does both) |
| Copied upstream code | add `mymodel/_vendor/` (`relgnn`, `relgt`) |
| Own glue around vendored code | beside `_vendor/`, not inside it (`relgt/tokenize.py`) |

`__init__.py` re-exports the public names; importing it is what registers the
model. `models/lightgbm/` is the smallest complete example to copy.

One folder may register **several** models: `dummy` → `constant-global` +
`constant-per-entity`, `relgnn` → `relgnn` + `relgnn-es`, `tabpfn_rel` →
`tabpfn-rel-local` + `tabpfn-rel-client`.

```python
import numpy as np
from relbench.base import Database, EntityTask, Table

from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace

MYMODEL_SPACE = SearchSpace(space=_config_space(), default_overrides={})


@register_model(search_space=MYMODEL_SPACE)
class MyModel(RelArenaModel):
    name = "mymodel"

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None: ...

    def predict(
        self, task: EntityTask, db: Database, table: Table
    ) -> np.ndarray: ...  # shape (len(table),)
```

## 2. Inputs and outputs — the datatypes

A RelBench entity task is a relational database plus one label table per split.
**One row of a label table is one prediction**: an entity, an anchor timestamp,
and (on train/val) the target. The task's temporal semantics ask for a prediction
as of that timestamp. RelArena enforces the database cutoff for each evaluation
phase; the model decides whether and how to enforce the finer per-anchor cutoff
inside that database.

| Argument | Type | Contents |
| --- | --- | --- |
| `task` | `relbench.base.EntityTask` | `entity_col`, `entity_table`, `time_col`, `target_col`, `task_type`, `timedelta`, `metrics` |
| `db` | `relbench.base.Database` | `table_dict: dict[str, Table]` — the whole (censored) database |
| `train_table`, `val_table`, `table` | `relbench.base.Table` | one row per (entity, timestamp) |
| return of `predict` | `np.ndarray` | shape `(len(table),)` |

A `Table` is a pandas frame plus its schema: `df`, `pkey_col`,
`fkey_col_to_pkey_table: dict[str, str]` (the foreign keys, and which table each
points at), `time_col`. Traversing the relational structure means following
`fkey_col_to_pkey_table` between frames in `db.table_dict`.

A label table's columns are `[task.time_col, task.entity_col, task.target_col]`;
`task.timedelta` is the forward window the label was computed over.
`task.task_type` is `BINARY_CLASSIFICATION` or `REGRESSION` — those are the only
two in scope (`ENTITY_TASK_TYPES` in [`tasks.py`](../src/relarena/tasks.py); link
prediction and multilabel are excluded).

### Nested temporal validation — the two phases

Each phase is a `Split` ([`dataset.py`](../src/relarena/dataset.py)) bundling a
censored database with the label tables that phase is allowed to see:

| Phase | Class | `db` censored at | fit on | predict on | scored against |
| --- | --- | --- | --- | --- | --- |
| Tuning | `InnerSplit` | `val_timestamp` | `train_table` | `eval_table` = val | `eval_target` — the val labels, passed explicitly |
| Final | `OuterSplit` | `test_timestamp` | `train_table`, plus `val_table` per `refit_on_full_data` | `eval_table` = **masked** test table | RelBench's own hidden test labels |

The outer split's `eval_table` has the target column stripped, and the split
object carries no test labels at all. This prevents test-label leakage through
the model-facing evaluation interface. The inner split's val labels *are*
visible and can be used at the developer's discretion: in particular, a method may
use the validation labels for tuning, for training, or for both (see
[3e. Refit on full data](#3e-refit-on-full-data)).

Full write-up: [`temporal-validation.md`](temporal-validation.md).

### The contract — what is fixed

Defined in [`src/relarena/model.py`](../src/relarena/model.py), with the temporal
split construction in [`dataset.py`](../src/relarena/dataset.py).

| Rule | Why |
| --- | --- |
| `predict` returns shape `(len(table),)` | what `EntityTask.evaluate` expects. Regression: the value. Binary: probability of the positive class. Wrapping an sklearn-style estimator → call `predict_to_contract` (`_shared/predict_contract.py`), which also handles the positive-class-absent case |
| `val_table` is `None` on the refit (see [3e](#3e-refit-on-full-data)) | supplied while tuning (early stopping, checkpoint selection); `None` when the selected config is refit on train+val, since there is no held-out split then. Methods have the choice to refit on train+val, but do not have to |
| `db` is censored at the phase boundary | rows after the validation or test boundary are unavailable, and test labels are withheld. A model may additionally restrict every example to rows at or before its anchor timestamp. Ignoring that finer temporal structure cannot reveal test labels or advance the database beyond the phase boundary |
| `seed` must make the run reproducible | same seed + same config → same predictions |

### Class attributes

| Attribute | Values | Default | Who deviates |
| --- | --- | --- | --- |
| `name` | unique string | — | required |
| `supported_task_types` | `ENTITY_TASK_TYPES`, or a narrower frozenset | all entity task types | `graphsage`, `relgt` → binary + regression only. Runner refuses out-of-set tasks rather than producing a meaningless number |
| `refit_on_full_data` | `True` / `False` | `True` | `relgt`, `relgnn-es`, `rdblearn` → `False` |
`refit_on_full_data` is a genuine modelling decision, not just a flag — see
[3e](#3e-refit-on-full-data).

### Model or system?

RelArena accommodates end-to-end systems through a separate contract instead
of forcing them into the model tuning loop. Implement `RelArenaSystem` when the
entry owns the complete procedure that produces its final predictions: its own
hyperparameter search, training-budget selection, component choices, ensemble,
or pretrained recipe. Register it with `@register_system`; systems do not
declare a `SearchSpace`.

```python
import numpy as np
from relbench.base import EntityTask

from relarena.dataset import InnerSplit, OuterSplit
from relarena.registry import register_system
from relarena.system import RelArenaSystem


@register_system
class MySystem(RelArenaSystem):
    name = "mysystem"

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        ...
        return predictions  # aligned with outer_split.eval_table
```

The harness passes the actual split objects, not a separate system-input
wrapper. A system usually uses `inner_split` to make its internal choices and
`outer_split` to produce the reportable predictions, but the API does not
prescribe that algorithm: a system may ignore the inner split.

Systems and models are not the same kind of result. Both receive the same
censored data and hidden-label test protocol, but a model's score isolates one
method under the shared tuning pipeline while a system's score credits its
whole procedure. A system therefore produces one `SystemResult` with final test
metrics and total runtime. It does not produce fake configs, a placeholder
validation score, or an `n_trials` value. Result frames mark the row with
`kind="system"` so leaderboards can report models alone or both populations.

A system still obeys the protocol boundaries: it only receives the censored
databases in the split objects, `outer_split.eval_table` is masked, and the
outer split contains no test labels. The framework validates the returned
shape and owns final evaluation.

#### Shared end-to-end runtime constraint

Systems have the same runtime allowance as models. The current policy sets a
24-hour ceiling for the complete run on the largest tasks, including
preprocessing. Methods should not aim to exhaust that allowance; most methods
finish in under 12 hours even on the largest tasks. For a model, the total
covers every tuning trial, selection, final fit, and prediction. For a system,
it covers all work used to produce the returned predictions, including internal
search, component training, refitting, and ensembling. Moving selection inside
`run` does not move it outside the runtime budget.

`SystemResult.time_total` measures the call to `RelArenaSystem.run`. If a
submission performs preprocessing separately through a public cache warmer,
that time is reported and counted toward the same end-to-end allowance even
though it is not part of `time_total`. The `time_limit` argument is a soft
budget; a system must either honor it or document why its training loop cannot
stop safely at an arbitrary wall-clock boundary. See
[The tuning regime](tuning-regime.md) for the full runtime policy.

The two execution paths are available directly as `run_model_experiment` and
`run_system_experiment`. `run_experiment` is the shared dispatcher used by the
CLI when it only has a registered method class.

A parameter-free space does not by itself make a system: `constant-global` runs
one config under the shared pipeline and stays a model. The test is whether the
score can be attributed to the method itself under the harness's controls.

## 3. Tuning

### How a run works

`run_experiment` ([`runner.py`](../src/relarena/runner.py)) is three phases:

1. **Tune** — `plan_configs` turns the registered space into an ordered list of
   `(tag, config)`; `tune` ([`tuner.py`](../src/relarena/tuner.py)) runs one trial
   per config on the inner split: instantiate `model_cls(config)`, `fit` on train,
   `predict` on val, score with the task's primary metric. A trial that raises is
   recorded as failed, not fatal to the run.
2. **Select** — `select_best` picks the best validation score, respecting the
   metric's direction.
3. **Refit** — the selected config is fit on the outer split according to
   `refit_on_full_data` and predicts test. The `"default"`-tagged config is refit
   the same way, so **every run reports both a default and a tuned number**.

One model instance = one config; `self.config` is the resolved dict. The config
equal to `default_overrides` gets the tag `"default"`, the rest `r0, r1, ...`.

The budget (`--n-trials`), the runtime policy, what we ran for the released
baselines, and why making tuning comparable is hard are covered in
[tuning-regime.md](tuning-regime.md). The TL;DR of a permissible tuning regime:

- At most 24h total runtime per dataset (pre-processing included), measured on
  the largest dataset. This does not mean that this limit should be exhausted:
  all current baselines except RelGT stay under 12h even on the largest dataset.
- Somewhat larger search spaces on smaller datasets are permissible within
  reason, but not encouraged. Current implementations mostly use constant-sized
  search spaces.
- For a `fixed_grid`, the budget is the grid size — run everything, default
  first. For a sampled space, the number of evaluated configs will depend on
  `n_trials`. Make a choice according to the runtime constraints and what seems
  reasonable.
- Unsure whether a regime is reasonable? Open an issue and ask or use the
  other implemented methods for reference.

### The choices you make

The space is not a method on the class; `@register_model(search_space=...)` binds
the two in the registry (mirrors AutoGluon / TabArena). See
[`src/relarena/search_space.py`](../src/relarena/search_space.py).

| # | Decision | Options |
| --- | --- | --- |
| 3a | Form of the space | `space` (sampled) / `fixed_grid` (enumerated) / neither |
| 3b | Grid order vs. budget | order is priority — the cap truncates |
| 3c | The default regime | `{}` (library defaults) / an explicit dict |
| 3d | Static or task-dependent | a `SearchSpace` / a `TaskStats` factory |
| 3e | The final fit | refit on train+val / keep val held out |

### 3a. Form of the space — at most one of `space` / `fixed_grid` (both raises)

| Form | Semantics | Use when | Users |
| --- | --- | --- | --- |
| `space=ConfigurationSpace(...)` | sampled randomly, seeded by the run seed | continuous or large spaces | `lightgbm` (14 params, one a `Constant`), `graphsage`, `relgnn` |
| `fixed_grid=[cfg, ...]` | enumerated in order, capped at `n_trials` | genuinely small + discrete | `rdblearn` (TFM × depth), `tabpfn_rel`, `relgt` |
| neither | only the default config runs | parameter-free baseline | `constant-global`, `constant-per-entity` |

### 3b. Grid order and the budget cap

| Form | Configs evaluated | Does the default run? |
| --- | --- | --- |
| `space` | the default plus `n_trials` samples | always (it is prepended) |
| `fixed_grid` | the first `n_trials` entries | only if it sits early enough in the grid |

A grid is read in order and cut at the budget, so order is priority: put
`default_overrides` first, or a small budget drops it and the default regime is
never reported. The existing grids all lead with their default (the depth grids
ascend from the default depth 2; `relgt`'s grid puts its default
`{L=4, dropout=0.3}` first) and state the ordering rationale in a comment; do
the same. `SearchSpace` warns when `default_overrides` is missing from a grid
entirely, and again when the budget cuts entries.

### 3c. Form of `default_overrides` (the zero-tuning regime, reported/tagged as default)

| Form | Meaning | Users |
| --- | --- | --- |
| `{}` | pass no config; run the underlying library's own defaults | `lightgbm`, `constant-global` |
| explicit dict | a published default to reproduce | `graphsage`, `relgnn`, `relgt`, and the grid-based models (must be a grid entry) |

`relgnn` is the hard case: the paper hand-tunes a separate config per (dataset,
task) and publishes no single default, so ours is the modal per-task config
across the 21 tasks. When a paper has no single default, pick one and justify it
in the docstring. This is what lets a "we didn't tune it" number sit next to a
tuned one in the same table.

### 3d. Static space or task-dependent?

| Registered value | Resolution | Users |
| --- | --- | --- |
| a `SearchSpace` | used as-is | everything except `relgt` |
| `Callable[[TaskStats], SearchSpace]` | harness resolves once per task, when stats are known | `relgt` |

`TaskStats` currently carries only `num_train_nodes`. `relgt` uses it in three
tiers: above 1M training nodes a single config, above 100k a dropout sweep at
the default depth, below that the full L×dropout grid; each tier is a subset of
the next-larger one. Reserve the factory for scale-driven changes like this. A
space that varies by task for any other reason makes the cross-task comparison
harder to read.

### 3e. Refit on full data

The class attribute `refit_on_full_data` decides what the reported test number
comes from, i.e. whether the validation labels end up in the training set:

| Value | Final fit | val labels used for | Who |
| --- | --- | --- | --- |
| `True` (default) | selected config refit on train ∪ val, with `val_table=None` | config selection, then training | everything else |
| `False` | train on `train_table` alone, `val_table` passed as the monitoring set | config selection, then early stopping / checkpoint selection | `relgt`, `relgnn-es`, `rdblearn` |

`True` uses all the available data and keeps selection leakage-free, so it is
the default. Pick `False` only when the method's published protocol reports a
best-val checkpoint rather than a train+val refit.

One consequence of `True`: a model that early-stops has no held-out split on
the refit, because `val_table` is `None` there. It must fall back to a fixed
budget, e.g. the iteration count found while tuning.

`relgnn` registers both protocols as separate models. The better-performing
`relgnn-es` is the paper-facing RelGNN result; `relgnn` remains available as an
experimental full-data-refit variant and is excluded from the default release
leaderboard.

## 4. Pre-processing cache

Methods with expensive CPU-bound pre-processing may compute their artifacts
before a benchmark run and reuse them through a cache. This is permitted for
pragmatic reasons described in [tuning-regime.md](tuning-regime.md), but the
cache must remain an optimization of the normal data-loading and model pipeline.

Existing cache-owning families provide these examples:

| Family | Owned artifact | Runnable warmer |
| --- | --- | --- |
| shared DFS (`tabpfn-rel`, `rdblearn`) | Parquet matrices + depth maps | `python -m relarena.featurization.warm_cache` |
| RelGNN | materialized graph directory | `python -m relarena.models.relgnn.warm_cache` |
| RelGT | HDF5 token sequences | `python -m relarena.models.relgt.warm_cache` |

### Policy requirements

- We expect developers to make reasonable choices w.r.t. what is included in
  the pre-processing vs. the model training. In particular, the pre-processing
  should not be used to circumvent `relarena`'s data loading. This means that
  data should still be loaded using `RelBenchDatasetTask` objects via
  `inner_split()` / `outer_split()` methods. In general, we encourage
  developers to keep as much logic as possible within the actual `relarena`
  model.
- Cache pre-computation scripts are required to be public, so that (i) people
  can actually pre-compute the cache themselves and (ii) reviewers can check
  for oversights such as label leakage.
- Providing pre-built cache files for the user's convenience is permissible and
  appreciated, but the scripts to re-populate them must ship regardless, to
  facilitate community trust.

### Optional experimental caching API

`relarena` provides an optional, experimental caching API that can be useful
when implementing a local cache. It handles common mechanics such as miss
policies, safe keys, private scratch computation, and atomic publication. A
method may instead implement its cache independently without using this API;
the policy requirements above still apply.

When using the API, the preprocessing implementation remains responsible for
its keys, versioning, serialization, validation, and warmer. Keys should
describe the inputs that actually determine the artifact, excluding model or
training settings that do not affect it. Content fingerprints and explicit
preprocessing versions can be used to invalidate artifacts when their inputs or
meaning change.

See [`relarena.cache`](../src/relarena/cache.py) for the API and its design
notes, and [`tests/fixtures/cached_model.py`](../tests/fixtures/cached_model.py)
for a compact end-to-end example.

## 5. Where code goes

| Kind | Location | Examples |
| --- | --- | --- |
| Model-specific | inside the model folder | `tabpfn_rel/features.py` |
| Shared within a model family | `models/_shared/<family>/` | `_shared/gbdt/lgb.py`, `_shared/tfm/tfm.py`, `_shared/gnn/{graph,training,graph_cache}.py` |
| Shared across families | `models/_shared/` top level | `predict_contract.py` (serves `constant-global` + the TFM models) |
| Relational DB → flat feature table | `src/relarena/featurization/` | `entity` (RelBench LightGBM recipe), `dfs` (multi-hop, with depth cache) |
| Preprocessing cache mechanics | `src/relarena/cache.py` | local atomic publication of caller-owned artifacts |

If two models need the same helper, it moves to `_shared/`; it never stays in
one model for the other to import from. The layout exists to stop one model
reaching into another's private code.

## 6. Optional dependencies

Anything not already a core dependency becomes an extra and is imported lazily
inside `fit`:

```toml
[project.optional-dependencies]
mymodel = ["some-heavy-package>=1.0"]
```

Importing `relarena.models` imports every wrapper to register it, so a
module-level heavy import would make the whole registry require your extra.
Registration must work without it: the discovery scan skips a model whose
third-party dependency is missing and logs at info, but re-raises anything else.

Reuse an existing extra where the stack matches:

| Extra | Contents | Models |
| --- | --- | --- |
| `lightgbm` | LightGBM | `lightgbm` |
| `rdblearn` | DFS deps (TFM is core) | `rdblearn`, `tabpfn-rel-local` |
| `tabpfn-rel-api` | DFS + `tabpfn-client` | `tabpfn-rel-client` |
| `rdl` | shared RDL stack: PyG, PyTorch Frame, text embedder | umbrella, not used directly |
| `graphsage` / `relgnn` / `relgt` | `relarena[rdl]` (+ `einops`, `h5py` for `relgt`) | the GNN baselines |

The GNN baselines additionally need PyG's temporal (disjoint) neighbor sampling,
which requires `pyg-lib` (plus `torch-scatter` / `torch-sparse`; `torch-sparse`
alone errors). These are not part of any extra: the right wheel depends on the
machine's torch/CUDA build (Linux/GPU only), so install the matching wheels from
the [PyG index](https://data.pyg.org/whl/) on the target GPU machine.

## 7. Vendoring upstream code

Copied code goes in the model's `_vendor/` and needs all three of:

1. A per-file docstring naming the upstream repo, the pinned commit, and exactly
   what was modified (`relgnn/_vendor/model.py` is the pattern).
2. The path in `.coveragerc`'s `omit` list; vendored code is not ours to test
   and should not count against coverage.
3. An entry in `NOTICE` describing the upstream revision and modifications, with
   the upstream license text added to `src/relarena/models/VENDORED-LICENSES`.

2 and 3 are the ones that get forgotten. Keep `_vendor/` for genuinely copied
files only.

## 8. Tests

Test files mirror the source tree in location and name, dropping the `models/`
prefix: `models/mymodel/model.py` is tested by `tests/mymodel/test_model.py`.
Every test directory needs an `__init__.py`; several models share the
`test_model.py` basename, which collides under pytest's import mode otherwise.
Name tests `test__<feature>__<condition>__<expected>`.

A model's tests assert three things: it registers under its name, its default
config is a valid point in its space, and `predict` returns the contract shape.
The existing per-model tests are the template.

```bash
uv sync --extra mymodel                 # else your tests silently skip
OMP_NUM_THREADS=1 uv run pytest         # env var required on macOS

uv run pre-commit run --all-files
```

## 9. Checklist

- [ ] `models/mymodel/{__init__,model}.py`, registered with a unique `name`
- [ ] `predict` returns `(len(table),)`
- [ ] `default_overrides` set; for a `fixed_grid`, early enough to survive the cap
- [ ] Compute knobs (batch size, epochs, steps) fixed outside the space
- [ ] `supported_task_types` narrowed if the model cannot do a task type
- [ ] `refit_on_full_data` matches the method's published protocol
- [ ] Use `RelArenaSystem` + `@register_system` if the method owns its selection
      (see [Model or system?](#model-or-system))
- [ ] The complete model or system procedure stays within the shared runtime
      allowance, including separately run preprocessing
- [ ] Heavy imports inside `fit`, `predict`, or `run`; extra declared in `pyproject.toml`
- [ ] Shared helpers in `_shared/`, not imported from a sibling model
- [ ] Vendored code in `_vendor/` with docstring + `.coveragerc` + `NOTICE` + license text
- [ ] Expensive CPU pre-processing goes through a public, label-free cache-warm
      script
- [ ] `tests/mymodel/{__init__,test_model}.py`
- [ ] `uv run pre-commit run --all-files` clean, from the repo root

## Appendix — what every existing model chose

| Model | Folder | Space form | `default_overrides` | `refit_on_full_data` | Task types | Extra | Shared code used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `constant-global`, `constant-per-entity` | `dummy/` | neither | `{}` | `True` | all | core | `predict_contract` |
| `lightgbm` | `lightgbm/` | `space`, 14 params | `{}` | `True` | all | `lightgbm` | `featurization/entity`, `_shared/gbdt/lgb` |
| `rdblearn` | `rdblearn/` | `fixed_grid`, TFM × depth | `{tfm: tabpfn-v2, max_depth: 2}` | `False` | all | `rdblearn` | `featurization/dfs` + cache, `_shared/tfm` |
| `tabpfn-rel-local`, `tabpfn-rel-client` | `tabpfn_rel/` | `fixed_grid` (one space each) | knobs + `max_depth: 2` | `True` | all | `rdblearn` / `tabpfn-rel-api` | `featurization/dfs` + cache, `_shared/tfm` |
| `graphsage` | `graphsage/` | `space` | explicit | `True` | binary, regression | `graphsage` | `_shared/gnn/{graph,training,_vendor/gnn}` |
| `relgnn` (experimental) | `relgnn/` | `space` | explicit (modal per-task) | `True` | all | `relgnn` | `_shared/gnn`, own `_vendor/` |
| `relgnn-es` (paper-facing RelGNN) | `relgnn/` | `space` (same as `relgnn`) | explicit | `False` | all | `relgnn` | as `relgnn` |
| `relgt` | `relgt/` | `Callable[[TaskStats], SearchSpace]` → `fixed_grid` | explicit | `False` | binary, regression | `relgt` | `_shared/gnn`, own `_vendor/` + `tokenize.py` |

`rt-plurel` is a `RelArenaSystem`, so model search-space and final-fit columns
do not apply. It supports binary classification and regression and uses the
`rt` extra.
