# The tuning regime

How `relarena` makes tuning comparable across methods, what we actually ran for
the released baselines, and where the current regime falls short. Implementing a
model and its search space is covered in
[adding-a-model.md](adding-a-model.md); this document is about the budgets and
policy those spaces run under.

## Making tuning comparable is very hard

Making the tuning of methods comparable broadly comes down to two factors:

- (a) standardizing how tuning works (i.e., what data or metric do I use for
  tuning, preventing dataset-specific search-space/configs etc.)
- (b) standardizing how much methods tune (i.e., how many GPU hours do you
  spend to tune your method)

We believe that in the current version of `relarena`, we have mostly solved
(a), but are still far off solving (b), even though we believe that we made
some critical progress that allows us for the first time to compare methods,
while acknowledging the work that still has to be done to make them perfectly
comparable.

One deliberate exception to (a): entries registered as **systems**
(`RelArenaModel.kind`, see
[adding-a-model.md](adding-a-model.md#model-or-system)) run their own selection
inside `fit` instead of the standardized tuning pipeline. They stay inside (b)
— the same runtime policy applies — and inside every protocol rule (splits,
censoring, leakage guards), so their final scores are fairly earned; what is
lost is the attribution that (a) buys for models. Leaderboards keep the two
populations separable rather than pretending a system's score isolates a
method.

## Why is (b) so hard?

This comes mainly down to two factors:

- _Methods have very different runtimes:_ The variance among relational
  learning methods' runtimes is generally much higher than what we see in
  tabular learning. For example, on its slowest task (`rel-avito/user-visits`)
  `relgt` runs for over 40h on an RTX-6000 Pro GPU, while tuned `lightgbm`
  completes the same task in under 30 seconds. In this initial version of `relarena`, we
  tried to stay close to running the tuning regime as set out by the authors,
  if (i) it was provided and (ii) it was permissible within the `relarena`
  framework.
- _CPU-bound pre-processing:_ Most more advanced methods significantly benefit
  from running CPU-based pre-processing, which is cached on disk before the
  actual method runs. When starting this project, we first tried to go the hard
  route of implementing all methods as end-to-end baselines, which need to run
  both the pre-processing and the method itself during the run. We had to
  abandon this plan, as the pre-processing is generally CPU-bound and can take
  hours per dataset (`rdblearn`, `tabpfn-rel`, `relgnn(-es)`, `relgt`). It is
  quite rare to find compute nodes that possess both very strong
  CPUs and GPUs.
  The GPU nodes available to us (and to many industry practitioners) had only
  comparatively weak CPUs, which would have further increased the runtime.
  Given the scale at which we are running `relarena`, spending 100s of hours of
  CPU pre-processing on GPU nodes would have been unjustifiable in terms of
  cost (and would also be so for many practitioners or researchers). In
  addition, a genuinely end-to-end timed baseline would need to rebuild its
  artifacts rather than rely on a previously warmed store, meaning that any issue
  discovered in an implementation would require rerunning the entire
  pre-processing. We conclude that most current approaches to relational
  learning (including `tabpfn-rel` on large datasets) are unsuited to being run
  as an end-to-end model on a single piece of hardware.

Given this, we have made the pragmatic choice to allow for CPU-bound
pre-processing before running the model. `relarena` provides an optional,
experimental caching API that eases common implementation concerns such as
cache keys, miss policies, private scratch computation, and atomic publication.
Methods may use that API or implement caching independently.

The API does not make cache warming part of a `relarena` experiment: public
pre-computation scripts still run separately, typically on CPU-oriented
hardware, before the timed model run. Consequently, the runtime fields recorded
by `relarena` cover tuning, prediction, and final fitting, but not the time spent
building preprocessing artifacts for DFS or other cached methods. This remains
a blind spot in the current runtime comparison, even though the API makes such
caches easier to implement and reuse. The operational guidance and requirements
for cache pre-computation are documented in
[adding-a-model.md — Pre-processing cache](adding-a-model.md#4-pre-processing-cache).

## Rough runtime constraints

Given the issues mentioned above, we currently only provide some rough
constraints w.r.t. (b) to developers and rely on their goodwill to make
reasonable choices when implementing their method's tuning regime. We encourage
developers to open issues if they have questions regarding how to best set up
their tuning regime, and will provide our opinion on what constitutes a
reasonable tuning regime, if necessary. Given that `relarena` is currently in
the alpha-phase, we would like to have an open discussion with the community on
what regimes they consider reasonable.

**Current policy.** For now, we would like to restrict methods to have a
runtime of at most 24h per task. This should include pre-processing and the
actual model run within `relarena`, even though we currently only directly
record the runtime inside `relarena`. Additionally, the 24h constraint is meant
for the largest tasks. While it is generally permissible to have somewhat
larger search spaces for smaller tasks, this increase in size should be
within reasonable bounds. Tuning your method for 24h on a `rel-f1` task is
definitely not reasonable. Currently, all methods but `relgt` use a fixed-size
search space per dataset, i.e. they don't adjust it to the dataset size. In
general, nearly all methods do not actually hit the 24h constraint, with nearly all of
them running for at most 12h of recorded runtime per task (excluding the
cached pre-processing). Only `relgt` and `rt-plurel` currently break the 12h
barrier, both excluding their pre-processing: `relgt` with up to 41h of pure
GPU time on `rel-avito/user-visits`, and `rt-plurel` with up to 16.3h on the
largest `rel-amazon` task.

## What we ran

**Why does `n_trials` differ between baselines?** The budget is not derived
from the search space: it is a per-run argument (`--n-trials`), so the numbers
in the table below are choices we made per method, and they were made for two
different reasons. Methods with a small discrete grid (`rdblearn`, the
`tabpfn-rel` variants and `relgt`) can run their whole grid once
the requested budget is large enough. For the methods with a sampled space
(`lightgbm`, `graphsage`, and RelGNN), the decision was mostly driven by compute
costs. Cheap methods like `lightgbm` get higher budgets, while GPU-bound methods
like `graphsage` and RelGNN get lower budgets.

Still, runtimes do differ significantly per method. For example, we could have
given `lightgbm` a much much larger search-space and it would still have been
much faster than the rest. Part of the solution to (b) will be even better
matching of compute across methods where appropriate.

**Why is RelGT currently an exception?** RelGT is the only RDL baseline that is
tuned and that actually provides the full details on how this tuning has been
done. For our initial version of `relarena` we tried to stay as close as
possible to RelGT's reported tuning regime, only making some minor adjustments.
As soon as we understand better how to solve (b), we plan to rerun all methods,
including RelGT, in that new and hopefully final tuning regime. For now, we
report `relgt`'s runtimes and note that its comparative performance might
suffer once we further standardize the tuning regime.

_Runtime stats derived from the current
[`results.csv`](../baseline_results/results.csv). These numbers do not include
the separately run CPU-bound pre-processing described above, which can take
hours (`rdblearn`, `tabpfn-rel`, RelGNN, `relgt`):_

| Model | `n_trials` | mean | min | p25 | p50 | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `constant-global` | 0 | 0.1 s | 0.0 s | — | 0.0 s | — | 0.3 s |
| `constant-per-entity` | 0 | 0.2 s | 0.0 s | — | 0.1 s | — | 1.3 s |
| `lightgbm` | 30 | 14 min | 0.1 min | 0.2 min | 2 min | 28 min | 53 min |
| `graphsage` | 4 | 47 min | 2 min | 9 min | 35 min | 83 min | 141 min |
| `rdblearn` | 6 | 11 min | 0.5 min | 2 min | 7 min | 11 min | 50 min |
| `tabpfn-rel-local` | 3 | 12 min | 0.6 min | 0.8 min | 4 min | 6 min | 85 min |
| `tabpfn-rel-client` | 3 | 76 min | 18 min | 72 min | 88 min | 94 min | 108 min |
| `relgnn-es` (RelGNN) | 10 | 73 min | 1 min | 6 min | 28 min | 99 min | 336 min |
| `relgt` | 9 | 511 min | 40 min | 175 min | 273 min | 466 min | 2442 min |
| `rt-plurel` | 0 | 351 min | 109 min | 129 min | 231 min | 516 min | 979 min |

Total runtime is per (dataset, task): all four recorded time columns summed
over every trial plus every recorded refit, with the distribution taken over the
21 RelBench v1 tasks.

The budget is set in exactly one place — `--n-trials` on the CLI (default 10),
or `PredictiveQuery.fit(n_trials=...)`. Nothing derives it from the search
space and no per-model default exists, so the numbers above are per-invocation
operator choices. Five caveats on reading them:

- `n_trials` is the requested argument, not a universal count of evaluated
  configurations. A sampled space evaluates its default plus `n_trials` random
  samples; a fixed grid evaluates at most its first `n_trials` entries; an
  untunable model evaluates only its default.
- `rt-plurel` is a system: its `n_trials` is 0, but its runtime *includes* the
  selection it runs inside each fit (the training-step search with in-loop
  validation and the context-configuration grid), so its row is not comparable
  to a model's single untuned fit.
- For `rdblearn` and the `tabpfn-rel` variants, the requested budget equals the
  grid size (`rdblearn` 6 = 2 TFMs × 3 depths; `tabpfn-rel` 3 = depths 2–4),
  meaning "run the whole grid" rather than a budget decision. The free choices
  are the random-search models: 30 for `lightgbm`, 4 for `graphsage`, and 10 for
  RelGNN (`relgnn-es`).
- For `relgt`, the requested budget is also not always what ran: it requested 9,
  but its `TaskStats` factory cuts the grid to 3 or 1 configs on larger tasks,
  so its effective budget is task-dependent whatever is passed.
- `constant-global`, `constant-per-entity`, and `lightgbm` ran on CPU. The
  local accelerated models ran on one `rtx-pro-6000`. `tabpfn-rel-client`
  records the CPU client device, but performs inference through the hosted API.
  The runtimes are therefore not a like-for-like hardware comparison.
