"""Final benchmark fitting and evaluation against hidden test labels."""

from __future__ import annotations

import time
from typing import Any, Type

from relbench.base import EntityTask

from relarena.core.cache import CacheConfig
from relarena.core.dataset import OuterSplit, concat_tables
from relarena.core.identity import RunIdentity
from relarena.core.metrics import evaluate_predictions, primary_metric
from relarena.core.model import RelArenaModel


def refit_and_evaluate(
    model_cls: Type[RelArenaModel],
    config: dict[str, Any],
    task: EntityTask,
    split: OuterSplit,
    *,
    seed: int,
    time_limit: float | None = None,
    cache: CacheConfig | None = None,
    run_identity: RunIdentity | None = None,
) -> dict:
    """Fit the selected `config` on the outer `split` and score it on `test`.

    Two final-fit regimes, chosen by `model_cls.refit_on_full_data`:

    * `True` (default): refit on the train+val union with `val_table=None` — no
      held-out split, so a model with early stopping falls back to a fixed budget.
    * `False`: train on train alone and pass `val` through, so a model that
      checkpoints on validation reports its best-val model (e.g. RelGT's protocol).

    Both train on the outer split's (test-censored) DB. The split carries no eval
    target (test labels are hidden), so we score with `target_table=None` and
    RelBench sources the unmasked test labels itself. Returns the test fields to
    attach to the trial.
    """
    metric = primary_metric(task)

    if model_cls.refit_on_full_data:
        train_table, val_table = concat_tables(split.train_table, split.val_table), None
    else:
        train_table, val_table = split.train_table, split.val_table

    model = model_cls(config, cache=cache, run_identity=run_identity)
    t0 = time.perf_counter()
    model.fit(
        task,
        split.db_state,
        train_table,
        val_table,
        seed=seed,
        time_limit=time_limit,
    )
    fit_time_refit = time.perf_counter() - t0

    t1 = time.perf_counter()
    test_pred = model.predict(task, split.db_state, split.eval_table)
    predict_time_refit = time.perf_counter() - t1

    test_metrics = evaluate_predictions(task, test_pred, None, metric)
    test_score = float(test_metrics[metric.__name__])
    return {
        "test_score": test_score,
        "test_metrics": test_metrics,
        "test_pred": test_pred,
        "fit_time_refit": fit_time_refit,
        "predict_time_refit": predict_time_refit,
    }


__all__ = ["refit_and_evaluate"]
