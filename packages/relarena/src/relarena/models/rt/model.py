"""`rt-plurel` — the Relational Transformer, fine-tuned per task from RT-P.

RT (https://arxiv.org/abs/2510.06377) is a relational *foundation* model: it
predicts directly over a database by attending over a sampled context of rows
drawn across foreign keys, so one pretrained net transfers to a schema it has
never seen. This wrapper runs the published fine-tuning recipe — a delta
fine-tune from RT-P, the PluRel-pretrained checkpoint the name refers to, every
value of it in [`config.py`](config.py) — on the harness's censored database.

**The two arms.** One system run receives both RelArena splits and executes
upstream's *selection* and *reporting* arms in sequence:

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
measured over, and the context directly between the two arms.

**Why the context is searchable at all.** Training draws each batch's context
shape at random from a cross-product of shapes rather than fixing one, so the
checkpoint is usable at any of them and the shape can be chosen after the fact,
with no retraining per configuration. The two halves only make sense together.

There is no harness search space: the step and context are selected internally,
without refitting once per candidate.

**Prediction.** After the outer arm, eight context seeds are evaluated, their
**raw** outputs averaged, and the sigmoid
(classification) or denormalization (regression) applied to the average — the
order upstream scores in. Predictions come back keyed by rustler node index, not
in table order, so they are scattered back to their rows through the split's
`node_idx_offset`.

The result uses the system schema: one real test prediction and the complete
runtime, with no dummy configuration or placeholder validation score.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from relbench.base import Database, EntityTask, Table, TaskType

from relarena.dataset import InnerSplit, OuterSplit, concat_tables
from relarena.identity import RunIdentity
from relarena.models.rt import config as cfg
from relarena.models.rt.export import TASK_DIR, preprocessed_dir, target_stats
from relarena.registry import register_system
from relarena.system import RelArenaSystem

logger = logging.getLogger(__name__)

#: The database name inside every exported `pre_dir`. RT addresses data as
#: `(db_name, task_name)`; both are fixed because the identity that matters
#: lives in the cache key, not in the exported tree.
DB_NAME = "relarena"

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
    """Delete this process's scratch directory."""
    global _SCRATCH
    if _SCRATCH is not None:
        _SCRATCH.cleanup()
        _SCRATCH = None


@register_system
class RTPluRelSystem(RelArenaSystem):
    """Relational Transformer, delta-fine-tuned per task from the PluRel RT-P.

    Named for its warm start: every arm begins from `stanford-star/rt-p`, the
    checkpoint pretrained under PluRel, and the per-task fine-tune is a delta on
    top of it. Nothing here is trained from scratch.
    """

    name = "rt-plurel"
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )

    def __init__(self, **kwargs: Any) -> None:
        """Instantiate the system for one complete inner-to-outer run."""
        super().__init__(**kwargs)
        # A local safetensors path, or a Hub spec when the warm start is what
        # validation chose (see `_fit_arm`). `from_pretrained` takes either.
        self._checkpoint: Path | str | None = None
        self._target_stats: tuple[float, float] | None = None
        self._task_type: TaskType | None = None
        # The split this instance trained on. `_predict` exports it beside the
        # table it scores: rustler normalizes a target by the *train* split's
        # statistics, so the scored split has to travel with the very rows the
        # model was fit on or it is denormalized by the wrong constants.
        self._train_table: Table | None = None
        #: The context configuration to predict under: chosen on val by the
        #: selection arm, carried to the reporting arm in `Selection`.
        self._context: tuple[int, int, int, bool] | None = None

    # -- complete system run ----------------------------------------------

    def run(
        self,
        task: EntityTask,
        *,
        inner_split: InnerSplit,
        outer_split: OuterSplit,
        seed: int,
        time_limit: float | None = None,
    ) -> np.ndarray:
        """Select on the inner split, refit on the outer split, and predict."""
        if time_limit is not None:
            logger.warning(
                "rt does not honor time_limit (%.0fs): the budget is the budget.",
                time_limit,
            )

        selection = self._fit_arm(
            task,
            inner_split.db_state,
            inner_split.train_table,
            inner_split.eval_table,
            phase=cfg.PHASE_INNER,
            seed=seed,
        )
        outer_train = concat_tables(outer_split.train_table, outer_split.val_table)
        self._fit_arm(
            task,
            outer_split.db_state,
            outer_train,
            None,
            phase=cfg.PHASE_OUTER,
            seed=seed,
            selection=selection,
        )
        return self._predict(task, outer_split.db_state, outer_split.eval_table)

    def _fit_arm(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        phase: str,
        seed: int,
        selection: cfg.Selection | None = None,
    ) -> cfg.Selection | None:
        """Run one RT arm, returning the inner arm's choices."""
        self._task_type = task.task_type
        self._target_stats = target_stats(train_table, task)
        self._train_table = train_table

        splits = {"train": train_table}
        if val_table is not None:
            splits["val"] = val_table
        pre_dir = preprocessed_dir(
            db,
            task,
            splits,
            cache=self.cache,
            identity=self._identity_for(phase),
            scratch_root=_scratch() / "export",
            db_name=DB_NAME,
        )

        rows = len(train_table.df)
        if phase == cfg.PHASE_OUTER:
            if selection is None:
                raise RuntimeError("rt: reporting arm requires the inner selection.")
            if selection.step == 0:
                logger.warning(
                    "rt: validation chose step 0 -- fine-tuning never beat the "
                    "published checkpoint on val. Reporting %s zero-shot.",
                    cfg.warm_start(task.task_type),
                )
                self._checkpoint = cfg.warm_start(task.task_type)
                self._context = selection.context
                return None

        total_steps = (
            cfg.selection_steps()
            if phase == cfg.PHASE_INNER
            else self._refit_steps(rows, selection)
        )
        cutoff = cfg.context_cutoff(task, "val" if phase == cfg.PHASE_INNER else "test")
        out_dir = self._train(
            task,
            pre_dir,
            total_steps,
            seed,
            phase=phase,
            has_val=val_table is not None,
            db_cutoff=cutoff,
        )

        if phase == cfg.PHASE_INNER:
            if val_table is None:
                raise RuntimeError("rt: selection arm requires a validation table.")
            self._checkpoint, step = _best_checkpoint(out_dir, task.task_type)
            self._context = self._tune_context(task, pre_dir, val_table, seed)
            selected = cfg.Selection(step=step, rows=rows, context=self._context)
            logger.info(
                "rt: validation chose step %d of %d over %d rows.",
                step,
                total_steps,
                rows,
            )
            return selected

        self._checkpoint = _last_checkpoint(out_dir)
        return None

    def _identity_for(self, phase: str) -> RunIdentity | None:
        """Scope this run's cache identity to one RT arm."""
        return None if self.run_identity is None else self.run_identity.for_phase(phase)

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

    def _refit_steps(self, rows: int, selection: cfg.Selection) -> int:
        """The outer arm's budget: the chosen step, rescaled to the bigger split."""
        self._context = selection.context
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
        phase: str,
        has_val: bool,
        db_cutoff: int | None,
    ) -> Path:
        """Run one arm of the fine-tune; return its output directory."""
        from rt.train import main as train_main

        run_id = f"{phase}-seed{seed}-steps{total_steps}"
        out_root = _scratch() / "train"
        args = cfg.train_args(
            phase=phase,
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
            phase,
            total_steps,
            args["load_ckpt_path"],
        )
        train_main(**args)
        # `run_subdir(entity, project, run_id)` with entity=None.
        return out_root / "no-entity" / args["project"] / run_id

    # -- inference machinery, shared by test prediction and context search ---

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

    # -- test prediction ---------------------------------------------------

    def _predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Score the outer split's masked test table."""
        if self._task_type is None:
            raise RuntimeError("rt test prediction called before the reporting arm.")

        # Only the reporting arm gets this far, and it needs both: the
        # checkpoint to score with, and the split it was fit on, which travels
        # into the export because rustler takes its normalizing statistics from
        # the train split (see export.py).
        if self._checkpoint is None or self._train_table is None:
            raise RuntimeError("rt test prediction called before the reporting arm.")

        import torch
        from rt import RelationalTransformer

        # The masked test table is not handed to `_fit_arm`, so it is exported here —
        # beside the train split the model was fit on, which is what rustler
        # takes the normalizing statistics from (see export.py).
        split = "test"
        pre_dir = preprocessed_dir(
            db,
            task,
            {"train": self._train_table, split: table},
            cache=self.cache,
            identity=self._identity_for(cfg.PHASE_OUTER),
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


__all__ = ["RTPluRelSystem", "clear_scratch"]
