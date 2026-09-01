"""Turn RelArena results into a TabArena-style leaderboard via `bencheval`.

Two steps:

- `to_bencheval_frame` is the seam between RelArena's rich, score-based results
  (`results.summary_to_dataframe` / `results.csv`) and `bencheval`'s
  leaderboard input: a tidy long frame with one row per `(method, task)`, a
  lower-is-better `metric_error >= 0`, and train/infer times. Pure pandas.
- `compute_leaderboard` ranks that frame with `bencheval.BenchmarkEvaluator`.
  `bencheval` is the optional `leaderboard` extra, imported lazily inside the
  function so importing this module stays dependency-free.

The board's task set comes from the frame it is given (optionally narrowed by a
`subsets.TaskSubset`); methods that do not cover it are dropped.
"""

from __future__ import annotations

import logging

import pandas as pd

from relarena.evaluation.subsets import TaskMask, apply_subset
from relarena.metrics import to_metric_error

logger = logging.getLogger(__name__)


def to_bencheval_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Derive a bencheval leaderboard frame from a RelArena results frame.

    Maps the results schema onto bencheval's contract:

    - `method`       ← `model`
    - `task`         ← `f"{dataset}/{task}"` (bencheval keys ELO on `task`
      alone, and RelBench task names are not unique across datasets)
    - `metric_error` ← `to_metric_error(test_score, <primary metric>)`, using
      the per-row `metric` column (always a registered primary, never the
      auxiliary native-metric columns)
    - `time_train_s` ← `fit_time_tuning + fit_time_refit`
    - `time_infer_s` ← `predict_time_refit`

    Only the val-selected config per run contributes (one row per method/task);
    runs without a test score (failed or test-skipped) are dropped. The caller
    must still ensure the method × task matrix is dense before ranking —
    `bencheval` rejects a sparse matrix.
    """
    rows = results
    if "selected" in rows.columns:
        rows = rows[rows["selected"]]
    rows = rows[rows["test_score"].notna()]

    metric_error = [
        to_metric_error(score, metric)
        for score, metric in zip(rows["test_score"], rows["metric"], strict=True)
    ]
    frame = pd.DataFrame(
        {
            "method": rows["model"].to_numpy(),
            "task": (
                rows["dataset"].astype(str) + "/" + rows["task"].astype(str)
            ).to_numpy(),
            "metric_error": metric_error,
            "time_train_s": (
                rows["fit_time_tuning"].fillna(0.0) + rows["fit_time_refit"].fillna(0.0)
            ).to_numpy(),
            "time_infer_s": rows["predict_time_refit"].fillna(0.0).to_numpy(),
        }
    )
    # A NaN primary score (e.g. roc_auc on a degenerate window) survives the
    # test_score filter but cannot be a valid error — drop those rows too.
    return frame[frame["metric_error"].notna()].reset_index(drop=True)


def _drop_incomplete_methods(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop methods without a result on every task in the frame, logging them.

    `bencheval` requires a dense method × task matrix, and one row per
    (method, task) as `to_bencheval_frame` produces. Dropping the *methods*
    rather than the tasks keeps the task set fixed by the frame, so one
    method's partial coverage never shrinks the board the others are ranked on;
    narrow the frame with `subset=` to rank it on a set it does cover.
    """
    if frame.empty:
        return frame
    n_tasks = frame["task"].nunique()
    per_method = frame.groupby("method")["task"].nunique()
    dropped = per_method[per_method < n_tasks]
    if not dropped.empty:
        logger.warning(
            "Leaderboard: dropping %d method(s) without a result on all %d tasks: %s",
            len(dropped),
            n_tasks,
            ", ".join(f"{m} (missing {n_tasks - n})" for m, n in dropped.items()),
        )
    return frame[~frame["method"].isin(dropped.index)].reset_index(drop=True)


def method_kind(model: str) -> str:
    """`"model"` or `"system"` for a registered method; `"model"` if unregistered.

    A system faces the same tasks, splits, metrics, and budget as every model —
    the comparison is fair on final performance — but selects its own
    hyperparameters or components inside `fit`, so its score credits the whole
    package rather than isolating one method (see `RelArenaModel.kind`).
    Unregistered names (reference baselines, retired methods) rank as models.
    """
    # Built-ins register on this import, which a leaderboard-only caller (load
    # a results CSV, rank it) has no other reason to have made. Without it the
    # registry is empty, every lookup falls through to "model", and a system
    # silently joins the models-only board.
    import relarena.models  # noqa: F401
    from relarena.registry import registry

    try:
        return registry.get(model).kind
    except KeyError:
        return "model"


def compute_leaderboard(
    results: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    baseline_method: str | None = "constant-global",
    kinds: frozenset[str] | None = None,
    subset: str | TaskMask = "all",
) -> pd.DataFrame:
    """Rank a RelArena results frame into a TabArena-style leaderboard.

    Builds the bencheval input with `to_bencheval_frame`, drops methods that
    did not run every task (see `_drop_incomplete_methods`), then ranks with
    `bencheval.BenchmarkEvaluator`. The board is sorted by normalized loss
    (`loss_rescaled` — per-task min-max error averaged across tasks) and also
    reports mean rank, win-rate, and ELO. ELO is anchored on `baseline_method`
    so its scale is stable across runs; the anchor is ignored if that method is
    absent, and ELO is skipped entirely for a single method (it scores pairwise
    comparisons, of which there are none). Returns an empty frame if no method
    covers the whole task set.

    `subset` narrows the board's tasks (see `relarena.evaluation.subsets`),
    applied before the density check so a method covering only that subset ranks
    here while the full board still drops it.

    Pass `reference` (a frame from `relarena.evaluation.load_reference_results`)
    to rank self-reported baselines alongside the reproduced runs as ordinary
    methods.

    `kinds` restricts the board by `method_kind`: `frozenset({"model"})` is the
    models-only board that isolates methods under the shared tuning pipeline;
    the default (`None`) ranks models and systems together, which is fair on
    final performance but does not attribute a system's score to any one part
    of its package. Publishing both, clearly labeled, is the intended use.
    """
    from bencheval.evaluator import BenchmarkEvaluator  # optional `leaderboard` extra

    if reference is not None:
        results = pd.concat([results, reference], ignore_index=True)
    if kinds is not None:
        results = results[results["model"].map(method_kind).isin(kinds)]
    results = apply_subset(results, subset)
    frame = _drop_incomplete_methods(to_bencheval_frame(results))
    if frame.empty:
        return pd.DataFrame()
    anchor = baseline_method if baseline_method in set(frame["method"]) else None
    # Elo raises on a single method rather than degrading, unlike the other
    # columns, which stay meaningful (or go NaN) when there is nothing to rank
    # against — a one-model sweep should still get a board.
    pairwise = frame["method"].nunique() > 1
    return BenchmarkEvaluator().leaderboard(
        frame,
        sort_by="loss_rescaled",
        include_rescaled_loss=True,
        include_winrate=True,
        include_elo=pairwise,
        include_improvability=False,
        baseline_method=anchor,
        elo_kwargs=(
            {"calibration_framework": anchor}
            if pairwise and anchor is not None
            else None
        ),
    )
