"""Every RT knob this model runs under, written at the call it belongs to.

Transcribed from upstream's fine-tuning submission
(`expts/fine_tune/submit.py` in rishabh-ranjan/relational-transformer, as of
`7c97204` for the recipe and `f098abe` for the mixed-context arm): batch 256, lr
5e-4 constant, weight decay 0.1, Muon, delta fine-tuning from RT-P, an EMA of
the weights at `swa_momentum=0.9999`, and an eval every 100 steps. Each batch
draws its context shape from a cross-product of shapes rather than fixing one,
which is what lets the shape itself be chosen after training, on val, by
inference alone (`context_grid`).

Upstream is one run: train on train+val, score test as it goes, keep the last
step. RelArena needs two — `PHASE_INNER` picks the budget on val, `PHASE_OUTER`
trains that budget on train+val and is what gets reported — so the differences
are the ones that split makes necessary, each marked `DEPARTURE` where it is
written:

1. **The inner arm evaluates on val, not test.** Upstream scores test as it
   trains — it is charting a curve, and `submit_ens.py` produces the reportable
   number afterwards over the whole split. RelArena must not: `task.evaluate`
   owns the test labels, and the model is never handed them. So the inner arm
   evaluates `val`, and the outer arm evaluates nothing at all
   (`eval_splits=[]`).
2. **`db_cutoff` is a timestamp, not a split name** (upstream: `"test"`).
   Upstream names a split and lets relbench resolve it against the release; a
   caller-assembled database has no release to ask, so `context_cutoff` reads
   the horizon off the split's own rows and passes the integer. It is set
   wherever a split is being *scored* — the val horizon on the inner arm, the
   test horizon in `predict` — and `None` on the outer arm's training, which
   scores nothing and exports no split but the one it trains on. This is not a
   duplicate of RelArena's censoring: that removed post-cutoff *database* rows,
   while this bounds how far a *context* may reach, and the label tables are
   exported beside the database. Getting it wrong is invisible in the output —
   see the README.
3. **The selection eval reads fewer rows than upstream's**: 1024 against
   upstream's `2**12`, at upstream's ensemble of 4. The row cap is what is
   given up to keep a dense curve affordable; the ensemble is not, because it
   is what separates two checkpoints whose val scores differ by less than one
   context draw's noise. `eval_freq=100` is upstream's, so the curve is dense,
   and a fixed shuffle seed makes it read the same rows every time. Which rows
   and which context draws are *not* upstream's: they are deliberately disjoint
   from the context search's, so the two val decisions do not compound on one
   sample (`step_shuffle_seed`, `step_context_seed`).
4. **The inner arm stops after `patience_steps()` without an improvement**
   (upstream: no patience). It is looking for a peak, not spending a budget.
   It also scores only the SWA net (`eval_live=False`), so the patience follows
   the net that gets reported rather than either of two.
5. **The context configuration is chosen on val, after training**
   (upstream: fixed). Training mixes context shapes, so the shape can be ranked
   afterwards by inference alone; `context_grid` is what is ranked, `tune_rows`
   and `tune_rows` / `selection_ensemble_size` are what it is ranked over,
   and `Selection.context` carries the winner to the reporting arm.
6. `wandb_disabled=True`, `targets={}`, `entity=None`, `project="relarena"`.

**How the budget is chosen and carried.** `rt.train` already does the picking:
it tracks the best-val step per (task type, net) and publishes
`best_swa_clf.safetensors` and friends, selecting on `BEST_METRICS = [("clf",
"auroc", max), ("reg", "nmae", min)]` — the metrics RelArena selects on too. The
inner arm runs `selection_steps()` over `train`, RT keeps the best-val
step, and the outer arm trains the same number of *passes over the data* on its
larger `train + val` split: `refit_steps` scales the chosen step by the row
ratio. Matching raw steps instead would quietly give the reported model less
training than the one validation chose.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from relbench.base import TaskType

#: Upstream's selection arm: train on train, evaluate on val, pick a step.
PHASE_INNER = "inner"
#: Upstream's reporting arm: train on train+val at the chosen budget, no eval.
PHASE_OUTER = "outer"


def warm_start(task_type: TaskType) -> str:
    """The published weights this task type fine-tunes from.

    One head per task type, each its own subdirectory of the release. Resolved
    through huggingface_hub, so the ordinary HF cache applies (warm it from a
    login node and set HF_HUB_OFFLINE=1 on compute nodes without egress).

    Also what `predict` loads when validation chose step 0 — i.e. when
    fine-tuning never beat the checkpoint it started from.
    """
    return (
        "stanford-star/rt-p/"
        + {
            TaskType.BINARY_CLASSIFICATION: "classification",
            TaskType.REGRESSION: "regression",
        }[task_type]
    )


# =========================================================================== #
# rt.train.main -- the fine-tune
# =========================================================================== #
def train_args(
    *,
    phase: str,
    task_type: TaskType,
    pre_dir: str,
    db_name: str,
    task_name: str,
    train_split: str,
    eval_split: str | None,
    db_cutoff: int | None,
    total_steps: int,
    out_root: str,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    """The complete argument list for `rt.train.main`, for one phase.

    That entry point has no defaults by design — every argument is part of the
    record of the run — so every one of them is written here.
    """
    selecting = phase == PHASE_INNER
    return dict(
        # -- model. Not free choices: these must match the published checkpoint,
        # and `embedder` also selects which text-embedding file the preprocessed
        # data has to carry (so `preprocess_args` embeds under the same name).
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        compile=True,
        materialize_attn_masks=True,
        # The loss is the one this task's metric is scored by, and the warm
        # start is that task type's head: one per type, each its own
        # subdirectory of the release. Resolved through huggingface_hub, so the
        # ordinary HF cache applies (warm it from a login node and set
        # HF_HUB_OFFLINE=1 on compute nodes without egress).
        loss_fn={TaskType.BINARY_CLASSIFICATION: "bce", TaskType.REGRESSION: "l1"}[
            task_type
        ],
        load_ckpt_path=warm_start(task_type),
        # -- data + optimization
        db_task_list=[(db_name, task_name)],
        train_splits=[train_split],
        pre_dir=pre_dir,
        tokens_per_gpu=tokens_per_gpu(),
        num_workers=num_workers(),
        prefetch_factor=2,
        # The shapes a batch is drawn from. Training mixes them rather than
        # fixing one, which is what makes the context searchable afterwards by
        # inference alone: a net trained at a single context is not usable at
        # another, so the search in `model._tune_context` would be measuring
        # configurations the checkpoint was never fit for. Every configuration
        # `context_grid()` ranks is drawn from these lists.
        ctx_size_list=[128, 256, 512, 1024],
        local_ctx_size_list=[128, 256, 512, 1024],
        bfs_width_list=[16, 64, 256],
        prefer_latest_list=[False, True],
        num_walks=1_000,
        walk_length=20,
        mask_prob_max=0.0,  # token masking during training: off
        items_per_task=1_000_000_000,  # per-task cap on the stream: none
        # A zero-initialized additive delta on frozen pretrained weights: the
        # update is identical to ordinary fine-tuning, but weight decay pulls
        # toward the pretrained weights rather than toward zero.
        delta_finetune=True,
        optimizer="muon",
        lr=5e-4,
        wd=0.1,
        lr_warmup_steps=0,  # no schedule: lr from the first step to the last
        lr_decay_steps=0,
        grad_norm_max=1.0,
        total_bs=total_bs(),  # global batch (summed over ranks; we run one)
        total_steps=total_steps,
        # DEPARTURE: upstream runs the whole budget (`None`). The selection arm
        # stops once ten consecutive evals -- 1000 steps at `eval_freq=100` --
        # have neither improved on nor matched the best val metric.
        #
        # "The best val metric" is either net's, not the one we report.
        # `rt.train`'s `consider()` walks both the live net and the SWA net and
        # returns a single improved flag, so `improved_at` -- and with it the
        # patience -- is refreshed when *either* moves, while `_best_checkpoint`
        # reads `best_swa_*` alone. The live net can therefore hold the arm open
        # long after SWA has peaked: on rel-f1/driver-top3 SWA's best was step
        # 100 and the live net kept improving to 1500, so the arm ran to 2500
        # instead of 1100 and over half of it bought nothing that gets reported.
        #
        # Left as upstream's, because it is only ever wasted time: what is kept
        # is `best_swa_*` chosen by value, so a longer arm can only give SWA
        # more chances to improve, never a worse reported step. A tie counts as
        # an improvement too (`better(v, cur) == v` holds on equality), which
        # refreshes patience on a flat metric rather than exhausting it.
        #
        # The reporting arm has no val split and so nothing to stop on;
        # `rt.train` asserts as much.
        early_stop_after_steps=patience_steps() if selecting else None,
        # An EMA of the weights with a ~10k-step horizon, saved and evaluated
        # beside the live net; upstream's stand-in for a learning-rate decay,
        # and upstream's pairing for mixed-context training. The schedule and
        # the patience are *not* taken from that arm -- `lr_warmup_steps`,
        # `lr_decay_steps` and `early_stop_after_steps` above are ours.
        swa_momentum=0.9999,
        seed=seed,
        mmap_populate=True,  # pre-fault into RAM, not cold-faulting per item
        timeout_per_item=10.0,
        # Upstream's cadence. It is what makes the selection curve dense; the
        # outer arm has nothing to evaluate, so nothing to schedule.
        eval_freq=100 if selecting else None,
        # `publish_best` copies the winner out of the periodic checkpoints
        # before they are pruned, so they only need to outlive the eval that
        # named them.
        keep_all_ckpts=False,
        vector_db_path=None,
        # NOT a duplicate of RelArena's censoring, which is why this is not
        # None. RelArena's `db` has had post-cutoff *database* rows removed;
        # this bounds the horizon a *context* may reach, and the label tables
        # are exported beside the database. Without it the bound is each seed
        # row's own timestamp, so a context may quote earlier rows of the split
        # being scored -- their labels included. Upstream says "val"/"test" and
        # lets relbench resolve it; a caller-assembled database has no release
        # to resolve against, so the timestamp is passed directly.
        db_cutoff=db_cutoff,
        resume_save_mins=20.0,
        # -- in-loop validation. The inner arm evaluates on val, and that is
        # what selects the checkpoint; the outer arm evaluates nothing, since
        # RelArena scores test itself and never hands the model test labels.
        eval_splits=[eval_split] if eval_split else [],
        eval_db_task_list=[(db_name, task_name)],
        eval_pre_dir=pre_dir,
        eval_tokens_per_gpu=2**18,
        eval_num_workers=num_workers(),
        eval_prefetch_factor=2,
        eval_num_walks=1_000,
        eval_walk_length=20,
        # DEPARTURE: 1024 rows, not upstream's 2**12. A cheap, noisy curve is
        # enough to pick a step; the full split is scored once, afterwards, by
        # RelArena. `eval_shuffle_seed` below fixes which rows, so the curve
        # reads the same ones at every eval.
        eval_items_per_task=1024,
        # The in-loop eval that picks the *step* stays at one fixed shape: it is
        # ranking checkpoints of one run, not context configurations.
        eval_ctx_size_list=[1024],
        # The step is timed across the two context configurations of
        # `step_select_grid`, not one, and the best metric among them counts as
        # the published val number. rt selects on the first entry of
        # `eval_lcs_bw_pl_grid` alone unless this is set.
        eval_ctx_lcs_bw_pl_grid=step_select_grid(),
        eval_mmap_populate=True,
        eval_shuffle_seed=step_shuffle_seed(),
        eval_context_seed=step_context_seed(),
        # Per configuration in `step_select_grid`, so the step is timed on two
        # four-draw averages rather than one.
        eval_ensemble_size=selection_ensemble_size(),
        # Score the SWA net alone. The live net is not reported -- selection
        # takes `best_swa_*` -- and scoring it cost half the in-loop eval to
        # produce a number nothing reads, while *also* feeding early stopping:
        # `rt.train` refreshes the patience when either net improves, so the
        # live net held runs open long after SWA had peaked (on
        # rel-f1/driver-top3, SWA's best was step 100 and the arm ran to 2500).
        # With this off, the patience follows the thing being selected.
        eval_live=False,
        eval_vector_db_path=None,
        eval_lcs_bw_pl_grid=[(1024, 256, False)],
        # -- logging. DEPARTURE: upstream logs to wandb against published-best
        # reference lines; a benchmark run reports through RelArena instead.
        run_id=run_id,
        targets={},
        project="relarena",
        entity=None,
        run_name=run_id,
        wandb_disabled=True,
        out_root=out_root,
    )


# =========================================================================== #
# rt.eval.build_evaluator -- inference, one evaluator per ensemble member
# =========================================================================== #
def eval_args(
    *,
    device: str,
    context_seed: int,
    num_rows: int,
    db_cutoff: int | None,
    context: tuple[int, int, int, bool],
    ctx_sizes: list[int] | None = None,
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    """The complete keyword arguments for `rt.eval.build_evaluator`.

    `tasks` and `pre_dir` are positional and passed by the caller;
    `context_seed` is one member's, from `ensemble_context_seeds()`; and
    `num_rows` is the size of the split being scored.

    `context` is one `(ctx, local_ctx, bfs_width, prefer_latest)` point of
    `context_grid()` — the search ranks the grid on val, and `predict` reports
    under the winner. `ctx_sizes` widens the build to serve several ctx sizes at
    once: the evaluator builds contexts at the largest and scores every smaller
    size off a prefix, so one build answers for all of them. That is what makes
    the search 30 builds rather than 60 — and since a build costs setup time
    before it reads a row, halving the builds nearly halves the search.

    `num_rows` rather than an `items_per_task` the caller computes, because the
    translation is not the identity. The evaluator runs

        n_batches = min(len(dataset), max(1, items_per_task // eval_bs))

    which *floors*: at 726 rows and a batch of 256 that is 2 batches, 512 rows,
    and the last 214 never scored. So the cap is rounded up to a whole number of
    batches here. Overshoot is free -- `len(dataset)` bounds it, and the sampler
    fills any spare slot as a phantom.
    """
    ctx, lcs, bw, pl = context
    ctx_size_list = sorted(ctx_sizes or [ctx])
    tokens_per_gpu = 2**18
    # `build_evaluator`'s own batch size, from the two values above.
    eval_bs = max(1, tokens_per_gpu // max(ctx_size_list))
    return dict(
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        device=device,
        ctx_size_list=ctx_size_list,
        local_ctx_size=lcs,
        bfs_width=bw,
        prefer_latest=pl,
        # Held equal to what the mixed-context training used.
        num_walks=1_000,
        walk_length=20,
        tokens_per_gpu=tokens_per_gpu,
        # Every row of the split — never a subsample. The harness scores the
        # array against the whole split, so a cap here is a wrong answer rather
        # than a cheaper one. (The *selection* subsample is a different thing:
        # `train_args`' `eval_items_per_task`.)
        items_per_task=math.ceil(num_rows / eval_bs) * eval_bs,
        num_workers=num_workers(),
        context_seed=context_seed,
        # Which rows a capped read sees. `0` is upstream's and is what `predict`
        # uses, where it only sets the order because every row is read anyway.
        # The context search overrides it -- see `tune_shuffle_seed`.
        shuffle_seed=shuffle_seed,
        mmap_populate=True,
        prefetch_factor=2,
        vector_db_path=None,
        # NOT a duplicate of RelArena's censoring, and the only thing standing
        # between a prediction and another row's label. RT is an in-context
        # learner: a context quotes labelled rows of the task, and by default it
        # may quote rows of the split being scored. RelBench hides the test
        # labels, so `export.py` writes a constant placeholder in their place --
        # which a context would otherwise quote, teaching the model that every
        # test row is a 0. `context_cutoff` puts the whole split past the bound,
        # so none of it is reachable. See `train_args` for the rest.
        #
        # This is the only guard. rt used to carry a narrower second one
        # (`train_only_fallback`, which confined the context's fallback tier to
        # the `Train` split); it was removed in rt 1.3.0 because this bound
        # already excludes every row it would.
        db_cutoff=db_cutoff,
    )


def ensemble_context_seeds(size: int | None = None) -> list[int]:
    """The context seed of each ensemble member the test prediction averages over.

    Only the outer arm predicts for real; the inner arm returns a constant (see
    `model.py`), so this is the test ensemble and nothing else.

    `rt.eval.member_context_seed` mixes the base seed with the member index
    rather than adding, so member *m* of one run and member *m+1* of the next
    are not the same draw.

    The raw model outputs are averaged over these seeds, and only then is the
    sigmoid / denormalization applied — the order upstream's `_emit_and_score`
    scores in, and the reason this is not "average the probabilities".
    """
    from rt.eval import member_context_seed

    # DEPARTURE: upstream ensembles 4 at every in-loop eval, and its reportable
    # number comes from a separate `submit_ens.py` pass over the whole split.
    return [member_context_seed(0, member) for member in range(size or 8)]


# =========================================================================== #
# rt.preprocess.one -- the tensor export (see export.py)
# =========================================================================== #
def preprocess_args(*, dataset: str, out_dir: str, embed: bool) -> dict[str, Any]:
    """The complete argument list for `rt.preprocess.one`.

    `embed=False` is what `export.py` passes: it runs rustler here and then does
    the embedding itself, so that an identical text list computed for a previous
    export can be reused instead of recomputed (see `export._embed`). Nothing
    else calls this with `embed=True`; the parameter exists so that what this
    function returns is what is actually handed to `rt.preprocess.one`.
    """
    return dict(
        dataset=dataset,
        out_dir=out_dir,
        # The checkpoint's embedder: RT reads the embedding file by this name.
        embedder="all-MiniLM-L12-v2",
        batch_size=512,
        skip_tasks=False,
        embed=embed,
        upload_repo=None,
        public=False,
        revision=None,
    )


def context_cutoff(task: "Any", split: str) -> int:
    """The horizon a context may reach while scoring `split`, epoch seconds.

    **relbench's own split timestamp**, which is what upstream means by
    `db_cutoff="val"` / `db_cutoff="test"`. `task.dataset` carries both, so the
    number is read from the object the harness already hands the model -- no
    harness change, and no inference from the data.

    Every arm names the split whose horizon bounds it, including the reporting
    arm's training pass, which scores nothing: its database is censored at
    `test_timestamp`, so that is the horizon its contexts may reach, and saying
    so is upstream's `db_cutoff="test"` exactly. The bound is inert there --
    rustler takes `min(target_ts, cutoff)` and every train+val row precedes
    `test_timestamp` on all 21 entity tasks (checked) -- but an inert bound
    stated is better than an absent one inferred, which read as a discrepancy
    between the arms every time anyone looked at it.

    An earlier version of this derived the horizon from the scored split's own
    rows, as `min(row timestamps) - 1`. That needed the `-1` and was wrong to
    need it. rustler's `past_bound` is `ts > bound`, so anchoring on the first
    row leaves that row's whole cohort quotable by every later seed unless the
    bound is nudged below it -- measured at +0.034 AUROC of leaked signal on
    rel-f1/driver-top3 val. The split timestamp has no such problem, because it
    is not one of the split's own row timestamps in the first place.

    It is also the more robust anchor. `min(row timestamps)` equals the split
    timestamp on 15 of the 18 non-amazon entity tasks and misses it by 60 days
    on the three rel-f1 ones, where the 30-day seed grid puts its first two
    points in the winter break and no race falls on them. Deriving a protocol
    boundary from which rows happen to exist is how that kind of gap becomes a
    silent off-by-one.
    """
    import pandas as pd

    stamp = {
        "val": task.dataset.val_timestamp,
        "test": task.dataset.test_timestamp,
    }[split]
    return int(pd.Timestamp(stamp).timestamp())


# =========================================================================== #
# The context search: a grid tuned on val by inference alone
# =========================================================================== #
# The other half of mixed-context training (see `train_args`' `_list`
# arguments). Because the checkpoint was trained across the whole shape space,
# the shape can be *chosen* after training, by scoring a slice of val once per
# configuration. Nothing is retrained per configuration, which is the only
# reason a 36-point search is affordable at all.
#
# Transcribed from upstream's pair: the training half from `submit.py` at
# f098abe (the last commit whose `_list` arguments carried more than one entry),
# the tuning half from `submit_hpo_ens.py` at 9b186e7^.


def context_grid() -> list[tuple[int, int, int, bool]]:
    """The `(ctx, local_ctx, bfs_width, prefer_latest)` configurations to rank.

    `ctx_size_list x lcs_bw_pl_grid`, minus the combinations with
    `local_ctx_size > ctx_size`, which are not distinct from
    `local_ctx_size == ctx_size`. 60 configurations, and 30 context builds --
    every ctx size for one `(lcs, bw, pl)` is scored off a prefix of the same
    build, which is why the search costs 18 passes and not 36.

    The sizes are the ones training draws from (`train_args`), so every
    configuration ranked here is one the checkpoint was actually trained across.
    """
    return [
        (ctx, lcs, bw, pl)
        for lcs in (128, 256, 512, 1024)
        for bw in (16, 64, 256)
        for pl in (True, False)
        for ctx in (128, 256, 512, 1024)
        if lcs <= ctx
    ]


def step_shuffle_seed() -> int:
    """Which val rows the step search reads. Upstream's 0; see `tune_shuffle_seed`."""
    return 0


def step_context_seed() -> int:
    """The context-draw family the step search ranks checkpoints under.

    **Not** the family the context search ranks configurations under, and not
    the one `predict` reports from -- both of those are base 0.
    `member_context_seed` mixes the base through splitmix64, so a different base
    is a disjoint family rather than an offset into the same one: no draw is
    shared, where `base + member` would have shared all but one.

    Two selections on one val split should not also share their context
    randomness. A checkpoint that happens to look good under a particular set of
    context draws would otherwise be handed to a search that re-ranks
    configurations under those same draws, and the second decision cannot see
    that the first has already fitted them.

    The tuning keeps base 0 -- a prefix of the eight `ensemble_context_seeds`
    that `predict` averages over -- so the configuration is still ranked on
    draws from the family it will be used with. It is the *step* that moves.
    """
    return 1


def tune_shuffle_seed() -> int:
    """Which val rows the context search reads. **Not** the step search's seed.

    Both searches read a capped prefix of a shuffled val split, and until now
    both shuffled with the same seed: `step_shuffle_seed` for the step,
    this for the context. Same seed and both taking a prefix means the smaller
    read is a strict *subset* of the larger -- the 1024 rows that chose the step
    were 1024 of the 4096 that then chose the context. Two selections compounding
    on the same rows overfit them together, and the second cannot detect that the
    first already did.

    A different seed makes the draws independent instead of nested. It does not
    make them disjoint: expected overlap is `1024 * tune_rows() / n_val` rows,
    which is under 1% on the large val splits and unavoidably total on the small
    ones, where both reads take the whole split. That is the "up to availability"
    part -- on a 588-row val split there are no other rows to move to.
    """
    return 1


def tune_rows() -> int:
    """Val rows one tuning pass scores.

    Ranking 60 configurations against each other needs less of the split than
    the number being reported does.
    """
    return 2**12


def step_select_grid() -> list[tuple[int, int, int, bool]]:
    """The `(ctx, local_ctx, bfs_width, prefer_latest)` points the step is timed at.

    The two ends of the ctx range, each averaged over
    `selection_ensemble_size` context draws, and `rt.train` takes the **best**
    metric across them as the val improvement -- best in the metric's own
    direction, `max` for auroc and `min` for nmae.

    Two well-averaged endpoints rather than four single draws: a max over noisy
    estimates inflates in proportion to the noise, and inflated val selection is
    the failure this is meant to undo. The endpoints are also the informative
    ones -- intermediate shapes mostly interpolate, so a max would rarely come
    from them. A low ctx overfits sooner than a high one, so timing the step
    only at 1024 reports a checkpoint already past its peak whenever the context
    search then picks something small -- which it did on four of the first six
    tasks. Taking the best across shapes approximates choosing the step and the
    context jointly.

    Every point is in `context_grid()` and in what `train_args` draws from, so
    none of them is a shape the checkpoint has never seen.
    """
    return [
        (1024, 1024, 256, False),
        (128, 128, 16, True),
    ]


def patience_steps() -> int:
    """Steps without a val improvement before the selection arm stops.

    Ten times upstream-of-ours' 1000, because two things changed what
    "improvement" means. With both nets scored, `rt.train` refreshed the
    patience when *either* moved, and the live net moves on every step; with
    only the SWA net scored, the patience follows a weight EMA whose horizon is
    ~10k steps and which therefore improves in slower, coarser increments. The
    old number against the new signal would stop arms that were still climbing.

    An improvement is now "either config in `step_select_grid` beat its own
    best", not "their best-of beat the running best" -- a low ctx and a high ctx
    peak at different steps, so the patience should follow whichever is still
    climbing. That resets more often, so it buys more exploration per step of
    patience than the old semantics did.

    A hundred evals at `eval_freq=100`. Every arm measured so far peaked within
    2000 steps of its early stop, so this is 5x the largest gap observed.
    """
    return 10_000


def selection_ensemble_size() -> int:
    """Context seeds every val decision is made at: the step *and* the context.

    Upstream's four. Both things validation chooses are chosen from an average
    over this many context draws, so neither is ranked on a noisier quantity
    than the other, and neither is ranked on a much noisier quantity than the
    one reported (`ensemble_context_seeds`, eight).

    The *context* search's seeds are a prefix of the reported ensemble's, not a
    separate draw: it calls `ensemble_context_seeds(selection_ensemble_size())`,
    i.e. `member_context_seed(0, m)` for `m < 4`, the first four of the eight
    `predict` averages over -- so a configuration is ranked on draws from the
    family it will be used with. The *step* search draws from a disjoint family
    (`step_context_seed`), so the two decisions do not share their context
    randomness any more than they share their rows.

    It is not free. The in-loop eval runs every `eval_freq` steps over
    `eval_items_per_task` rows and this multiplies that work fourfold, on both
    the live and the SWA net. A single draw was the earlier choice and it made
    the step a coin-flip between checkpoints whose val scores differ by less
    than one context's worth of noise.
    """
    return 4


# =========================================================================== #
# The budget
# =========================================================================== #
def compile_inference() -> bool:
    """Whether the net is `torch.compile`d for scoring. It is for training.

    Two reasons, and the second is the one that belongs in this file.

    Speed: every inference stage loads the net once and then makes many forward
    passes through it -- `predict` is eight context seeds over the whole split,
    the context search is 72 evaluator builds -- so a compile warms up once and
    is amortized over all of them. The number of distinct graphs stays small
    because the shapes do: `eval_args` sizes its batch from `tokens_per_gpu //
    max(ctx_size_list)`, which is 256 for every build the search walks, and the
    sampler fills a short final batch with phantoms rather than reshaping it, so
    only the ctx size varies -- three variants at most, one for `predict`.

    Consistency: `train_args` sets `compile=True`, and a compiled graph is not
    bit-identical to an eager one. Scoring eagerly would mean the step and the
    context were chosen, and the number reported, under a slightly different
    numerical path from the one that trained the weights.

    Everything else the inference path wants is already upstream's:
    `rt.eval.evaluator` runs under `torch.inference_mode()` with `net.eval()`,
    the weights are cast to bfloat16 here, `materialize_attn_masks` defaults to
    True from the checkpoint config, and `mmap_populate` pre-faults the tensors
    so eight passes over one split do not re-fault it eight times.
    """
    return True


def total_bs() -> int:
    """The global batch size (summed over ranks; we run one). Upstream's."""
    return 256


def selection_steps() -> int:
    """The selection arm's step ceiling. DEPARTURE: upstream's is 25k.

    Not an epoch count: upstream trains every task the same number of steps --
    the task-size range spans a few thousand rows to millions, and a fixed step
    budget is the choice it makes about that -- and this keeps that shape.

    Affordable because mixed-context training is much cheaper per step than a
    fixed 1024 one: measured on rel-f1/driver-position, the selection arm ran
    1188 s under mixed contexts against 2073 s for the same budget at a fixed
    It is a ceiling, not a typical run. Every arm observed so far early-stopped
    between 1200 and 9500 steps -- but every one of those was observed under a
    *different* regime (both nets driving the patience, patience 1000, ensemble
    1, a coarser context grid), so none of them is evidence about where this
    configuration stops. The ceiling is set where it does not bind on anything
    we have reason to expect, rather than where the old observations sat.
    """
    return 50_000


def refit_steps(*, chosen_step: int, inner_rows: int, outer_rows: int) -> int:
    """The outer arm's budget: `chosen_step`, rescaled to the larger split.

    The inner arm chose a step over `train`; the outer arm trains on
    `train + val`. Handing it the same *step* would train the reported model on
    proportionally less data than the model validation chose, so the step is
    scaled by the row ratio, which holds the number of passes over the data
    fixed.
    """
    return max(1, math.ceil(chosen_step * outer_rows / max(1, inner_rows)))


# =========================================================================== #
# Machine-dependent throughput. No result depends on these.
# =========================================================================== #
def num_workers() -> int:
    """Dataloader workers: the cpus this process may actually run on.

    `os.cpu_count()` is the machine's, not the job's: under slurm a job holding
    14 of a node's 128 cpus would otherwise start 16 loader workers to share
    them, which torch warns about and which costs throughput rather than buying
    it. `sched_getaffinity` is the allocation.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not POSIX
        available = os.cpu_count() or 1
    return max(0, min(16, available - 1))


#: Override for `tokens_per_gpu()`, as a power of two or a plain integer.
#: Purely a throughput knob — it sets how much of a step's global batch fits in
#: one forward, not how large that batch is (`total_bs` is fixed at 256 and the
#: remainder is accumulated), so no reported number moves with it. The harness
#: has nowhere to put a machine-dependent value like this, hence an environment
#: variable rather than a config entry.
TOKENS_PER_GPU_ENV = "RELARENA_RT_TOKENS_PER_GPU"


def tokens_per_gpu() -> int:
    """Training tokens per forward per GPU: `2**18` on Blackwell, else `2**17`.

    A B200 holds twice what an A100 does, and filling it halves the number of
    gradient-accumulation micro-steps per optimizer step. Upstream picks the
    value from the slurm resource it asked for; here it is read off the device,
    which is the same choice made locally, and `TOKENS_PER_GPU_ENV` overrides
    both — for a node whose card is not what its capability implies, or to hold
    the value fixed while timing two machines against each other.
    """
    override = os.environ.get(TOKENS_PER_GPU_ENV)
    if override:
        return int(override)
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            return 2**18
    except Exception:  # pragma: no cover - absent torch/CUDA is not an error
        pass
    return 2**17


@dataclass(frozen=True)
class Selection:
    """What the selection arm chose, carried across to the reporting arm."""

    #: The best-val step `rt.train` published.
    step: int
    #: Rows in the split that step was measured over, so the reporting arm can
    #: rescale it to its own.
    rows: int
    #: The `(ctx, local_ctx, bfs_width, prefer_latest)` validation picked out of
    #: `context_grid()`. What `predict` reports under.
    context: tuple[int, int, int, bool]
    #: The checkpoint at that step, for the debug record. The reporting arm
    #: retrains to `step` on train+val rather than loading it.
    checkpoint: str = ""
