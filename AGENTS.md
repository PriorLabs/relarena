# relarena — agent notes

Start with the [README](README.md) for what the package is and how a run works.

## This repo is public

Everything committed here is published under Apache-2.0 — code, docs, and
`baseline_results/` alike. Nothing may reference Prior Labs–internal
infrastructure, systems, names, or paths. This applies to everything pushed
to the remote, not just tracked files: commit messages, branch names, and PR
text included.

## CI is the trust boundary

Changes under `.github/` need an approval from the `release-maintainers`
team that cannot be bypassed. Ordinary code review can be bypassed by
relational maintainers, so `.github/` is where secrets are guarded:

- A workflow step with access to a secret or an OIDC token may only run
  pinned actions or scripts under `.github/scripts/`. It must not run the
  package, the tests, or anything else outside `.github/`, since all of
  that would execute with the secret in its environment.
- Scripts under `.github/scripts/` are standalone: standard library or
  explicitly pinned dependencies, no imports from `relarena`.
- Scripts under `workflows/` are ordinary maintenance tooling under the
  normal review rules. Move a script into `.github/scripts/` before a
  secret-bearing step calls it.

Reviewers of a workflow change check every step that references
`secrets.*` or requests `id-token: write` against these rules.

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

- Import through the owning package's public API: shared functionality lives in
  `relarena_core`, benchmark functionality in `relarena`, and the TabPFN-Rel
  model in `tabpfn_rel`. Underscore-prefixed helpers are internal to their package.
- Respect the layering: model packages and `userdb/` import the generic core
  (`model`, `registry`, `search_space`, `tuner`, split types) and the shared
  infrastructure (`featurization/`) — never each other, and never
  the benchmark-only `evaluation/` subpackage. `models/_shared/` holds helpers
  for model families, including GNN and GBDT helpers. Keep the LightGBM fitting
  helper in `_shared/gbdt`; model-specific backend definitions belong with their
  model. Nothing outside `models/` imports `_shared/`.
- Keep generic and benchmark-specific code in separate modules: anything
  touching `relbench.datasets`/`relbench.tasks` or data checksums is benchmark
  code and doesn't belong in `model`/`tuner`/`search_space`.
- Heavy optional deps are lazy-imported inside `fit`, `predict` or `run` behind
  extras. Use core's `@register_model` and `@register_system` decorators; model
  imports register their own classes in the shared core registry. Baseline
  discovery imports public model packages automatically. Installed external
  model packages declare their modules in the `relarena.models` entry-point group.

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
OMP_NUM_THREADS=1 uv run --all-packages pytest        # otherwise: "Fatal Python error: Segmentation fault" in lightgbm
```

(There's no clean *permanent* local fix under `uv`: symlinking one `libomp` — e.g.
Homebrew's — over the wheels' bundled copies (`torch/lib`, `sklearn/.dylibs`) **does** stop
the segfault [verified], but `uv sync --all-packages` overwrites the symlinks, so it doesn't stick. The
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
