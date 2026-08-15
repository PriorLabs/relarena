# relarena — agent notes

Start with the [README](README.md) for what the package is and how a run works.

## Adding or changing a model

Read [docs/adding-a-model.md](docs/adding-a-model.md) first. Model development
is two things: the model folder (layout, fit/predict contract and its
datatypes, search-space choices, shared-code placement, optional dependencies,
vendoring, tests) and — for methods with expensive CPU pre-processing — a
public cache-warm script. `models/lightgbm/` is the smallest complete example
to copy.

## Refactor-friendly structure rules

Keep the contract/tuning core, models, predictive interface, and benchmark
evaluation layers cleanly separated:

- Import through the public API: external-facing names are re-exported in
  `relarena/__init__.py` (or a subpackage `__init__.py`), and docs, examples,
  and tests use those paths. Underscore-prefixed modules are internal and may
  move without notice.
- Respect the layering: model packages and `userdb/` import the generic core
  (`model`, `registry`, `search_space`, `tuner`, split types) and the shared
  infrastructure (`featurization/`) — never each other, and never
  the benchmark-only `evaluation/` subpackage. `models/_shared/` is for code
  shared between models; nothing outside `models/` imports it.
- Keep generic and benchmark-specific code in separate modules: anything
  touching `relbench.datasets`/`relbench.tasks` or data checksums is benchmark
  code and doesn't belong in `model`/`tuner`/`search_space`.
- Heavy optional deps are lazy-imported inside `fit` behind extras; no
  import-time side effects beyond explicit `register_model` calls.

## macOS: run tests / CLI with `OMP_NUM_THREADS=1`

`relarena` depends on both `torch` (a core dep) and `lightgbm`. On macOS
these bundle separate `libomp` runtimes, and if torch is loaded before lightgbm (which
happens whenever the model registry is imported) lightgbm **segfaults** — a known,
macOS-only AutoGluon/LightGBM issue
([autogluon#1442](https://github.com/autogluon/autogluon/issues/1442),
[LightGBM#6595](https://github.com/microsoft/LightGBM/issues/6595)), not a code bug. Linux
(including CI) is unaffected. So when running relarena tests or the CLI
locally on macOS, prefix with the env var:

```bash
OMP_NUM_THREADS=1 uv run pytest        # otherwise: "Fatal Python error: Segmentation fault" in lightgbm
```

(There's no clean *permanent* local fix under `uv`: symlinking one `libomp` — e.g.
Homebrew's — over the wheels' bundled copies (`torch/lib`, `sklearn/.dylibs`) **does** stop
the segfault [verified], but `uv sync` overwrites the symlinks, so it doesn't stick. The
durable single-`libomp` route is a conda-forge env, which this `uv`-managed repo doesn't
use. So `OMP_NUM_THREADS=1` is the practical local workflow — see the LightGBM FAQ if you
want to attempt the symlink route anyway.)

## `graphsage` extra (GNN baseline)

The `graphsage` model uses RelBench's GNN stack (`relbench.modeling.*`), pulled by the
`graphsage` extra: PyG + PyTorch Frame + a GloVe text embedder (`torch` is already core).
All heavy imports are lazy (inside `fit`), so registering the model — and the dep-free
registration/space/setup tests — work without the extra; the full `fit`/`predict` path is
exercised end-to-end on a GPU via the CLI smoke run, not in pytest. PyG **temporal (disjoint) neighbor sampling needs
`pyg-lib`** (+ `torch-scatter` / `torch-sparse`; `torch-sparse` alone errors), which
aren't in the extra (the right wheel depends on the machine's torch/CUDA build; Linux/GPU
only); install the matching wheels from the PyG index on the target GPU machine (see
[docs/adding-a-model.md](docs/adding-a-model.md#6-optional-dependencies)).
End-to-end runs want a GPU.

## `_MR` = model report (reference baselines)

Methods with an `_MR` suffix (e.g. `relgnn_MR`) are reported reference numbers,
not runs reproduced inside relarena; they load via
`relarena.evaluation.load_reference_results` and appear on a leaderboard or plot
only when explicitly passed as `reference=`. Per-method provenance and caveats
live in `baseline_results/SOURCES.md`.

## Other docs

- [docs/tuning-regime.md](docs/tuning-regime.md) — the tuning budgets and
  runtime policy, what the released baselines ran, and where the current
  regime falls short.
- [docs/temporal-validation.md](docs/temporal-validation.md) — the
  nested temporal-validation protocol and why the inner/outer splits exist.
- [docs/predictive-task.md](docs/predictive-task.md) — defining a predictive
  task over your own relational database.
