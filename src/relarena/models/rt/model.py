"""`rt-plurel` — the Relational Transformer, fine-tuned per task from RT-P.

RT (https://arxiv.org/abs/2510.06377) is a relational *foundation* model: it
predicts directly over a database by attending over a sampled context of rows
drawn across foreign keys, so one pretrained net transfers to a schema it has
never seen. This wrapper runs the published fine-tuning recipe — a delta
fine-tune from RT-P, the PluRel-pretrained checkpoint the name refers to, every
value of it in [`config.py`](config.py) — on the harness's censored database.

**The two arms.** Upstream's fine-tuning script runs a *selection* arm and a
*reporting* arm, and they are RelArena's two phases exactly:

  * **inner** — train on `train` for `config.selection_steps()`, and let
    `rt.train`'s own in-loop validation pick the checkpoint: every 100 steps it
    scores 1024 val rows at ensemble 1 and keeps the best, publishing
    `best_swa_*`, stopping once 1000 steps pass without either net improving
    (patience watches both, the report reads SWA -- see `config.py`). The step
    it settled on is read back out of that checkpoint — the step the run
    *stopped* at is not the step it reports. Then, with that checkpoint fixed,
    the **context configuration** is chosen: 36 points of
    `config.context_grid()` are scored over a slice of val by inference alone,
    and the best under the task's primary metric wins.
  * **outer** — train on `train + val` at that step rescaled by the row ratio,
    with no in-loop evaluation at all, and report the last step, predicting
    under the context the inner arm chose.

Rescaled rather than copied, because the outer split is bigger: the same raw
step count would train the reported model on proportionally less data than the
model validation chose. `Selection` carries the step, the row count it was
measured over, and the context.

**Why the context is searchable at all.** Training draws each batch's context
shape at random from a cross-product of shapes rather than fixing one, so the
checkpoint is usable at any of them and the shape can be chosen after the fact,
with no retraining per configuration. The two halves only make sense together.

**Parameter-free from the harness's side.** There is no search space: what gets
tuned — the step and the context — is chosen inside a single fit, not by
refitting once per candidate, so nothing is drawn from a RelArena config.

**Prediction, and the val score this model does not report.** Only the outer arm
predicts: eight context seeds, their **raw** outputs averaged, and the sigmoid
(classification) or denormalization (regression) applied to the average — the
order upstream scores in. Predictions come back keyed by rustler node index, not
in table order, so they are scattered back to their rows through the split's
`node_idx_offset`.

The inner arm's `predict` returns **zeros**. The harness scores whatever
`predict` returns and files it as `val_score`, so that column is a placeholder,
**not a validation result** — read it as "not reported" and ignore it in any
leaderboard or plot. The reason is cost: on the largest tasks a real val pass is
a sixth to a quarter of the training arm it would be reporting on, and it
selects nothing — the step was chosen by `rt.train`'s in-loop eval and the
context by `_tune_context`, both before `predict` is ever called. The test score
is unaffected; it is a real 8-seed ensemble over the whole test split.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from relbench.base import Database, EntityTask, Table, TaskType

from relarena.model import RelArenaModel
from relarena.models.rt import config as cfg
from relarena.models.rt.export import TASK_DIR, preprocessed_dir, target_stats
from relarena.registry import register_model
from relarena.search_space import SearchSpace

logger = logging.getLogger(__name__)

#: The database name inside every exported `pre_dir`. RT addresses data as
#: `(db_name, task_name)`; both are fixed because the identity that matters
#: lives in the cache key, not in the exported tree.
DB_NAME = "relarena"

#: Nothing to tune: `rt.train` chooses the epoch budget inside a single fit.
RT_SPACE = SearchSpace(default_overrides={})

#: RT's own name for each task type, as its checkpoint files are named.
_RT_TASK_TYPE = {
    TaskType.BINARY_CLASSIFICATION: "clf",
    TaskType.REGRESSION: "reg",
}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_offset(pre_dir: Path, split: str) -> int:
    """Rustler's global node index of `split`'s first row.

    `node_idx - offset` is the row's position in the parquet we exported, which
    is how a prediction finds its way back to the table row it belongs to (the
    evaluator does not yield rows in table order). Read from the preprocessed
    `table_info.json` rather than through `rt.eval`'s private helper of the same
    name — it is a data file with a stable shape, and reading it here keeps this
    wrapper off RT's internals.
    """
    info = json.loads((pre_dir / DB_NAME / "table_info.json").read_text())
    capitalized = {"train": "Train", "val": "Val", "test": "Test"}[split]
    key = f"{TASK_DIR}:Db" if f"{TASK_DIR}:Db" in info else f"{TASK_DIR}:{capitalized}"
    return int(info[key]["node_idx_offset"])


def _best_checkpoint(out_dir: Path, task_type: TaskType) -> tuple[Path | str, int]:
    """The selection arm's best-val checkpoint, and the step it was written at.

    `rt.train` tracks two nets and publishes three names per task type:
    `best_live_{tt}`, `best_swa_{tt}`, and `best_{tt}` — the better of the two
    on val. This takes the SWA net: upstream's recipe uses the weight EMA in
    place of a learning-rate decay, so those are the weights the recipe reports,
    and fixing the net means the reported model is the same *kind* of object on
    every task. Only the step is selected on val, never the net.
    """
    from safetensors import safe_open

    tt = _RT_TASK_TYPE[task_type]
    first, second = f"best_swa_{tt}", f"best_{tt}"
    path = out_dir / f"{first}.safetensors"
    if not path.exists():
        # The step-0 eval runs before the SWA tracker has averaged anything, so
        # there is no SWA net to save at it. If step 0 is *also* the best val
        # score the arm ever sees -- a fine-tune that only ever hurt -- no
        # `best_swa_*` is ever written. `best_{tt}` is then the live net at step
        # 0, i.e. the published checkpoint unchanged, which is the honest thing
        # to report for a task where fine-tuning did not help.
        fallback = out_dir / f"{second}.safetensors"
        if fallback.exists():
            logger.warning(
                "rt: no %s in %s (validation never improved past step 0, where "
                "SWA has averaged nothing); falling back to %s.",
                path.name,
                out_dir,
                fallback.name,
            )
            path = fallback
        else:
            # With `eval_live=False` there is no live checkpoint to fall back
            # to, and no SWA checkpoint exists at step 0 -- the tracker has
            # averaged nothing yet, so `swa_steps=0.safetensors` is never
            # written. Both missing therefore means validation never improved
            # on step 0, i.e. fine-tuning only ever hurt. The honest report is
            # the warm start itself, which is what step 0 *means*, and the
            # outer arm already knows to serve it unmodified.
            logger.warning(
                "rt: no %s or %s in %s -- validation never improved past step "
                "0. Reporting %s zero-shot.",
                path.name,
                fallback.name,
                out_dir,
                cfg.warm_start(task_type),
            )
            return cfg.warm_start(task_type), 0
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    step = int(metadata["step"])
    return path, step


def _last_checkpoint(out_dir: Path) -> Path:
    """The outer arm's final SWA weights: nothing selects, so it is the last step."""
    path = out_dir / "latest_swa.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"rt: no {path.name} in {out_dir}.")
    return path


#: The inner arm's choice, per `(dataset, task, seed)`, read by the outer arm.
#: The harness builds a *fresh model instance* for the refit, so the budget
#: validation picked cannot ride along on `self`; and it cannot ride in the
#: config either, since a parameter-free model's config is empty by definition.
#: Module level is therefore the only place left — which also means the two
#: phases must run in one process, as `run_experiment` does.
_SELECTED: dict[tuple[str, str, int], cfg.Selection] = {}

#: Scratch that has to outlive a single call (the tensor exports when no cache
#: is configured, and every training output directory). Process-lifetime, and
#: cleaned by `clear_scratch`.
_SCRATCH: tempfile.TemporaryDirectory | None = None


def _scratch() -> Path:
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = tempfile.TemporaryDirectory(prefix="relarena-rt-")
    return Path(_SCRATCH.name)


def clear_scratch() -> None:
    """Drop the carried selections and delete this process's scratch directory."""
    global _SCRATCH
    _SELECTED.clear()
    if _SCRATCH is not None:
        _SCRATCH.cleanup()
        _SCRATCH = None


@register_model(search_space=RT_SPACE)
class RTPluRelModel(RelArenaModel):
    """Relational Transformer, delta-fine-tuned per task from the PluRel RT-P.

    Named for its warm start: every arm begins from `stanford-star/rt-p`, the
    checkpoint pretrained under PluRel, and the per-task fine-tune is a delta on
    top of it. Nothing here is trained from scratch.
    """

    name = "rt-plurel"
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )
    #: The reporting arm trains on train+val, at the budget the selection arm
    #: chose. There is no held-out split left, and none is needed.
    refit_on_full_data = True
    #: A system, not a model: the step, the context configuration, and the
    #: warm start's whole training recipe are chosen inside `fit`, outside the
    #: harness's tuning pipeline. The score is the package's, and how much of
    #: it belongs to the architecture rather than the selection machinery is
    #: not identifiable from the benchmark alone. Implemented through the
    #: experimental system workarounds of the model API: all tuning inside a
    #: single fit of the parameter-free default config, and the selection
    #: carried to the refit phase via a module-level global (`_SELECTED`).
    kind = "system"

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        """Instantiate the model for a single hyperparameter `config`."""
        super().__init__(config, **kwargs)
        # A local safetensors path, or a Hub spec when the warm start is what
        # validation chose (see `fit`). `from_pretrained` takes either.
        self._checkpoint: Path | str | None = None
        self._target_stats: tuple[float, float] | None = None
        self._task_type: TaskType | None = None
        self._phase: str = cfg.PHASE_OUTER
        # The split this instance trained on. `predict` exports it beside the
        # table it scores: rustler normalizes a target by the *train* split's
        # statistics, so the scored split has to travel with the very rows the
        # model was fit on or it is denormalized by the wrong constants.
        self._train_table: Table | None = None
        #: The context configuration to predict under: chosen on val by the
        #: selection arm, carried to the reporting arm in `Selection`.
        self._context: tuple[int, int, int, bool] | None = None

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Run whichever arm this phase is (see the module docstring).

        `val_table` decides: present is the selection arm, absent the reporting
        arm. That is the harness's own signal for which phase it is in, so
        nothing here has to be told.

        `time_limit` is not honored. RT's loop has no wall-clock cutoff, and
        cutting it short would report a model trained for fewer epochs than the
        run it is recorded as. (The selection arm's early stopping is a
        different thing: it ends a search that has stopped improving, and never
        changes which step is reported.)
        """
        if time_limit is not None:
            logger.warning(
                "rt does not honor time_limit (%.0fs): the budget is the budget.",
                time_limit,
            )

        self._task_type = task.task_type
        self._target_stats = target_stats(train_table, task)
        self._train_table = train_table
        self._phase = self._phase_of(val_table)

        splits = {"train": train_table}
        if val_table is not None:
            splits["val"] = val_table
        pre_dir = preprocessed_dir(
            db,
            task,
            splits,
            cache=self.cache,
            identity=self.run_identity,
            scratch_root=_scratch() / "export",
            db_name=DB_NAME,
        )

        rows = len(train_table.df)
        if self._phase == cfg.PHASE_OUTER:
            selection = _SELECTED.get(self._selection_key(seed))
            if selection is not None and selection.step == 0:
                # Validation preferred the published checkpoint to every
                # fine-tuned step it saw. Training zero steps of it is not a
                # thing `rt.train` can be asked for, and one step is not what
                # was chosen -- so the reported model is the warm start itself,
                # unmodified, which is exactly what "step 0" means.
                logger.warning(
                    "rt: validation chose step 0 -- fine-tuning never beat the "
                    "published checkpoint on val. Reporting %s zero-shot.",
                    cfg.warm_start(task.task_type),
                )
                self._checkpoint = cfg.warm_start(task.task_type)
                # The context the inner arm chose still governs `predict`.
                # `_refit_steps` is what normally carries it across the arms,
                # and this path does not reach it.
                self._context = selection.context
                return
        total_steps = (
            cfg.selection_steps()
            if self._phase == cfg.PHASE_INNER
            else self._refit_steps(rows, seed)
        )

        # One rule, both arms: bound the context at the horizon of the phase.
        # The selection arm's database is censored at `val_timestamp` and it
        # scores `val`; the reporting arm's is censored at `test_timestamp`.
        # Naming both is upstream's `db_cutoff="val"` / `"test"` exactly.
        cutoff = cfg.context_cutoff(
            task, "val" if self._phase == cfg.PHASE_INNER else "test"
        )
        out_dir = self._train(
            task,
            pre_dir,
            total_steps,
            seed,
            has_val=val_table is not None,
            db_cutoff=cutoff,
        )

        if self._phase == cfg.PHASE_INNER:
            self._checkpoint, step = _best_checkpoint(out_dir, task.task_type)
            self._context = self._tune_context(task, pre_dir, val_table, seed)
            _SELECTED[self._selection_key(seed)] = cfg.Selection(
                step=step,
                rows=rows,
                checkpoint=str(self._checkpoint),
                context=self._context,
            )
            logger.info(
                "rt: validation chose step %d of %d over %d rows.",
                step,
                total_steps,
                rows,
            )
        else:
            self._checkpoint = _last_checkpoint(out_dir)

    def _phase_of(self, val_table: Table | None) -> str:
        """Which arm this `fit` call is: the harness's own phase label.

        `run_identity.phase` is what the runner sets (`"inner"` / `"outer"`),
        which is more robust than reading `val_table is None`: the two agree
        under `refit_on_full_data=True`, but the phase label is the harness's
        own statement of which arm this is rather than an inference from the
        shape of the call.
        """
        phase = None if self.run_identity is None else self.run_identity.phase
        if phase in (cfg.PHASE_INNER, cfg.PHASE_OUTER):
            return phase
        # No identity (a direct caller, not the runner): fall back to the shape
        # of the call, which is right for the default regime.
        return cfg.PHASE_INNER if val_table is not None else cfg.PHASE_OUTER

    def _tune_context(
        self, task: EntityTask, pre_dir: Path, val_table: Table, seed: int
    ) -> tuple[int, int, int, bool]:
        """Rank the context grid on val by inference; return the winner.

        Nothing is retrained: the checkpoint step-selection just chose is scored
        once per configuration over a capped slice of val, and the best under
        the task's own primary metric wins. Affordable only because the net was
        trained across the whole shape space -- see `config.train_args`.

        The grid is walked by **context build**, not by configuration. An
        evaluator builds contexts at the largest ctx size it is given and scores
        every smaller size off a prefix of the same build, so one build answers
        for every ctx size sharing a `(local_ctx, bfs_width, prefer_latest)` --
        24 builds for 60 configurations. A build costs setup time before it reads a
        row, so walking configurations instead would nearly double the search.
        """
        import torch
        from rt import RelationalTransformer
        from rt.data import get_tasks
        from rt.eval import build_evaluator

        from relarena.metrics import is_better, primary_metric

        metric = primary_metric(task)
        truth = val_table.df[task.target_col].to_numpy()
        cap = min(cfg.tune_rows(), len(val_table.df))
        cutoff = cfg.context_cutoff(task, "val")
        device = _device()
        offset = _seed_offset(pre_dir, "val")
        # Loaded once, not once per configuration: 36 loads of an 85M-parameter
        # net is minutes of nothing.
        net = RelationalTransformer.from_pretrained(
            str(self._checkpoint), device=device, compile=cfg.compile_inference()
        ).to(torch.bfloat16)
        rt_tasks = get_tasks(str(pre_dir), [(DB_NAME, TASK_DIR)], ("val",))
        seeds = cfg.ensemble_context_seeds(cfg.selection_ensemble_size())

        builds: dict[tuple[int, int, bool], list[int]] = {}
        for ctx, lcs, bw, pl in cfg.context_grid():
            builds.setdefault((lcs, bw, pl), []).append(ctx)

        total: dict[tuple[int, int, int, bool], np.ndarray] = {}
        count: dict[tuple[int, int, int, bool], np.ndarray] = {}
        n = len(val_table.df)
        for (lcs, bw, pl), ctx_sizes in builds.items():
            sizes = sorted(ctx_sizes)
            for context_seed in seeds:
                args = cfg.eval_args(
                    device=device,
                    context_seed=context_seed,
                    num_rows=cap,
                    db_cutoff=cutoff,
                    context=(max(sizes), lcs, bw, pl),
                    ctx_sizes=sizes,
                    # Not the seed the step search read with, so the two
                    # selections do not compound on one set of rows.
                    shuffle_seed=cfg.tune_shuffle_seed(),
                )
                evaluator = build_evaluator(rt_tasks, str(pre_dir), **args)
                for result in evaluator.evaluate_raw(
                    [(net, "")], sizes, with_node_idxs=True
                ):
                    _t, ctx_size, _labels, preds_by_prefix, _n, node_idxs = result
                    key = (int(ctx_size), lcs, bw, pl)
                    rows = np.asarray(node_idxs, dtype=np.int64) - offset
                    if key not in total:
                        total[key] = np.zeros(n, dtype=np.float64)
                        count[key] = np.zeros(n, dtype=np.int64)
                    total[key][rows] += np.asarray(
                        preds_by_prefix[""], dtype=np.float64
                    )
                    count[key][rows] += 1

        best_score: float | None = None
        best_context = None
        for key, sums in total.items():
            seen = np.flatnonzero(count[key])
            if seen.size == 0:
                continue
            raw = sums[seen] / count[key][seen]
            pred = (
                raw * self._target_stats[1] + self._target_stats[0]  # type: ignore[index]
                if self._task_type == TaskType.REGRESSION
                else 1.0 / (1.0 + np.exp(-raw))
            )
            score = float(metric(truth[seen], pred))
            if best_score is None or is_better(score, best_score, metric):
                best_score, best_context = score, key
        logger.info(
            "rt-plurel: %s wins over %d configs in %d builds (%s %.4f on %d val rows)",
            best_context,
            len(total),
            len(builds) * len(seeds),
            metric.__name__,
            best_score if best_score is not None else float("nan"),
            cap,
        )
        return best_context

    def _selection_key(self, seed: int) -> tuple[str, str, int]:
        """What the inner arm's choice is filed under for the outer arm."""
        identity = self.run_identity
        dataset = "direct" if identity is None else identity.dataset
        task_name = (
            "direct" if identity is None or identity.task is None else identity.task
        )
        return (dataset, task_name, seed)

    def _refit_steps(self, rows: int, seed: int) -> int:
        """The outer arm's budget: the chosen step, rescaled to the bigger split."""
        selection = _SELECTED.get(self._selection_key(seed))
        if selection is not None:
            self._context = selection.context
        if selection is None:
            logger.warning(
                "rt: no selection recorded for %s; refitting at the full %d "
                "steps. The selection arm has to run first, in this process.",
                self._selection_key(seed),
                cfg.selection_steps(),
            )
            return cfg.selection_steps()
        steps = cfg.refit_steps(
            chosen_step=selection.step, inner_rows=selection.rows, outer_rows=rows
        )
        logger.info(
            "rt: refitting %d steps (step %d over %d rows, rescaled to %d).",
            steps,
            selection.step,
            selection.rows,
            rows,
        )
        return steps

    def _train(
        self,
        task: EntityTask,
        pre_dir: Path,
        total_steps: int,
        seed: int,
        *,
        has_val: bool,
        db_cutoff: int | None,
    ) -> Path:
        """Run one arm of the fine-tune; return its output directory."""
        from rt.train import main as train_main

        run_id = f"{self._phase}-seed{seed}-steps{total_steps}"
        out_root = _scratch() / "train"
        args = cfg.train_args(
            phase=self._phase,
            task_type=task.task_type,
            pre_dir=str(pre_dir),
            db_name=DB_NAME,
            task_name=TASK_DIR,
            train_split="train",
            eval_split="val" if has_val else None,
            db_cutoff=db_cutoff,
            total_steps=total_steps,
            out_root=str(out_root),
            run_id=run_id,
            seed=seed,
        )
        logger.info(
            "rt: %s arm, %d steps from %s",
            self._phase,
            total_steps,
            args["load_ckpt_path"],
        )
        train_main(**args)
        # `run_subdir(entity, project, run_id)` with entity=None.
        return out_root / "no-entity" / args["project"] / run_id

    # -- inference machinery, shared by predict and the context search -------

    def _raw_predictions(
        self,
        net: Any,
        pre_dir: Path,
        split: str,
        table: Table,
        offset: int,
        device: str,
        cutoff: int | None,
        *,
        context: tuple[int, int, int, bool],
        seeds: list[int],
        cap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`(row indices, averaged raw outputs)` for `table` under one context.

        The raw model outputs are averaged over `seeds` before anything is
        applied to them, which is the order upstream scores an ensemble in.
        `cap` bounds how much of the split is read -- the whole thing when
        predicting, a slice when ranking configurations.
        """
        from rt.data import get_tasks
        from rt.eval import build_evaluator

        rt_tasks = get_tasks(str(pre_dir), [(DB_NAME, TASK_DIR)], (split,))
        total = np.zeros(len(table.df), dtype=np.float64)
        counts = np.zeros(len(table.df), dtype=np.int64)
        for context_seed in seeds:
            args = cfg.eval_args(
                device=device,
                context_seed=context_seed,
                num_rows=cap,
                db_cutoff=cutoff,
                context=context,
            )
            evaluator = build_evaluator(rt_tasks, str(pre_dir), **args)
            for result in evaluator.evaluate_raw(
                [(net, "")], args["ctx_size_list"], with_node_idxs=True
            ):
                _task, _ctx, _labels, preds_by_prefix, _n, node_idxs = result
                rows = np.asarray(node_idxs, dtype=np.int64) - offset
                if rows.size and (rows.min() < 0 or rows.max() >= len(table.df)):
                    raise RuntimeError(
                        f"rt: seed row indices out of range [{rows.min()}, "
                        f"{rows.max()}] for a table of {len(table.df)} rows."
                    )
                total[rows] += np.asarray(preds_by_prefix[""], dtype=np.float64)
                counts[rows] += 1
        seen = np.flatnonzero(counts)
        return seen, total[seen] / counts[seen]

    # -- predict -----------------------------------------------------------

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Score `table` — for real on the outer arm, as a placeholder on the inner.

        **The inner arm returns zeros.** Nothing downstream of it selects
        anything (the checkpoint was chosen inside `fit`, by `rt.train`'s in-loop
        validation over 1024 rows), and scoring the whole val split at ensemble 1
        costs up to a quarter of the training arm on the largest tasks. So the
        `val_score` the harness derives from this is a placeholder — see the
        module docstring. The outer arm's test prediction is a real 8-seed
        context ensemble over every row.
        """
        if self._task_type is None:
            raise RuntimeError("rt.predict called before fit.")

        if self._phase == cfg.PHASE_INNER:
            logger.info(
                "rt: skipping the val prediction (%d rows); rt reports no "
                "validation score, and nothing downstream selects on it.",
                len(table.df),
            )
            return np.zeros(len(table.df), dtype=np.float64)

        # Only the reporting arm gets this far, and it needs both: the
        # checkpoint to score with, and the split it was fit on, which travels
        # into the export because rustler takes its normalizing statistics from
        # the train split (see export.py).
        if self._checkpoint is None or self._train_table is None:
            raise RuntimeError("rt.predict called before fit.")

        import torch
        from rt import RelationalTransformer

        # The masked test table is not handed to `fit`, so it is exported here —
        # beside the train split the model was fit on, which is what rustler
        # takes the normalizing statistics from (see export.py).
        split = "test"
        pre_dir = preprocessed_dir(
            db,
            task,
            {"train": self._train_table, split: table},
            cache=self.cache,
            identity=self.run_identity,
            scratch_root=_scratch() / "export",
            db_name=DB_NAME,
        )
        offset = _seed_offset(pre_dir, split)
        # A prediction for one row of this split must not read another row's
        # label, so the context stops at the split's own horizon.
        cutoff = cfg.context_cutoff(task, "test")

        device = _device()
        model = RelationalTransformer.from_pretrained(
            str(self._checkpoint), device=device, compile=cfg.compile_inference()
        ).to(torch.bfloat16)
        rows, raw = self._raw_predictions(
            model,
            pre_dir,
            split,
            table,
            offset,
            device,
            cutoff,
            context=self._context,
            seeds=cfg.ensemble_context_seeds(),
            cap=len(table.df),
        )
        if rows.size != len(table.df):
            raise RuntimeError(
                f"rt: {len(table.df) - rows.size} of {len(table.df)} rows got no "
                "prediction; the evaluator did not cover the split."
            )
        out = np.empty(len(table.df), dtype=np.float64)
        out[rows] = raw
        raw = out
        if self._task_type == TaskType.REGRESSION:
            mean, std = self._target_stats  # type: ignore[misc]
            return raw * std + mean
        return 1.0 / (1.0 + np.exp(-raw))


__all__ = ["RT_SPACE", "RTPluRelModel", "clear_scratch"]
