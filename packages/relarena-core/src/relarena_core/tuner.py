"""Random-search and fixed-grid execution under a caller-supplied budget.

Key design points (vs. ad-hoc per-model tuning in baseline repos):
  * the **harness** decides the budget (`n_trials` / `time_limit_per_trial`),
    not the model; released runs use documented method-specific budgets;
  * the model's hyperparameter space is a registered `SearchSpace`, which the
    harness samples (the model itself declares no sampling);
  * every trial records its configuration, metrics, phase-split wall-clock times,
    and optionally its validation/test predictions as useful analysis metadata.

Callers select the best configuration with `relarena_core.selection.select_best`.
This module runs the trials and returns their records.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Type

from relbench.base import EntityTask

from relarena_core.cache import CacheConfig
from relarena_core.dataset import InnerSplit
from relarena_core.identity import RunIdentity
from relarena_core.metrics import evaluate_predictions, primary_metric
from relarena_core.model import RelArenaModel
from relarena_core.results import TrialResult, config_id_for
from relarena_core.search_space import (
    SearchSpace,
    SearchSpaceProvider,
    TaskStats,
    resolve_search_space,
)

logger = logging.getLogger(__name__)


def _concise_error(exc: BaseException) -> str:
    """One-line `Type: message (file:line)` for the result row (full trace logged).

    `file:line` is the innermost frame — where the exception was raised.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    where = ""
    if frames:
        last = frames[-1]
        where = f" ({os.path.basename(last.filename)}:{last.lineno})"
    message = " ".join(str(exc).split())  # collapse any newlines to keep one line
    return f"{type(exc).__name__}: {message}{where}"


def run_trial(
    model_cls: Type[RelArenaModel],
    config: dict[str, Any],
    config_tag: str,
    task: EntityTask,
    split: InnerSplit,
    *,
    seed: int,
    time_limit: float | None = None,
    cache_predictions: bool = True,
    cache: CacheConfig | None = None,
    run_identity: RunIdentity | None = None,
) -> TrialResult:
    """Fit one config on the (inner) `split` and score it on the split's eval table.

    The split's `eval_table` doubles as the model's early-stopping val during
    `fit` — this is the tuning phase, where the eval set is the held-out
    validation set. The test number is produced separately by
    a caller-owned final fit on the outer split.

    Never raises on model failure — a failed trial is returned with `error`
    set, so one bad config does not abort a sweep.
    """
    metric = primary_metric(task)
    base = dict(
        config=config,
        config_id=config_id_for(config),
        config_tag=config_tag,
    )

    try:
        model = model_cls(config, cache=cache, run_identity=run_identity)

        t0 = time.perf_counter()
        model.fit(
            task,
            split.db_state,
            split.train_table,
            split.eval_table,
            seed=seed,
            time_limit=time_limit,
        )
        fit_time_tuning = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_pred = model.predict(task, split.db_state, split.eval_table)
        predict_time_tuning = time.perf_counter() - t1
        val_metrics = evaluate_predictions(task, val_pred, split.eval_target, metric)
        val_score = float(val_metrics[metric.__name__])

        return TrialResult(
            **base,
            val_score=val_score,
            val_metrics=val_metrics,
            fit_time_tuning=fit_time_tuning,
            predict_time_tuning=predict_time_tuning,
            val_pred=val_pred if cache_predictions else None,
        )
    except Exception as exc:
        # Full traceback to the logs; a concise summary to the result row, so
        # failures are queryable from the CSV without bloat.
        logger.exception(
            "Trial failed (model=%s, config_tag=%s, config_id=%s)",
            model_cls.name,
            config_tag,
            base["config_id"],
        )
        return TrialResult(**base, error=_concise_error(exc))


def plan_configs(
    search_space: SearchSpace, n_trials: int, seed: int
) -> list[tuple[str, dict[str, Any]]]:
    """Build the ordered `(tag, config)` plan the tuner evaluates.

    Config generation is delegated to `SearchSpace.configs` (the
    `default_overrides` config plus `n_trials` random samples, or an explicit
    ordered fixed grid). The config equal to the space's `default_overrides` is
    tagged `"default"` (the runner reports it as the zero-tuning regime); the
    rest get `"r{i}"` tags.
    """
    configs = search_space.configs(n_trials, seed)
    plan: list[tuple[str, dict[str, Any]]] = []
    default_tagged = False
    for i, cfg in enumerate(configs):
        if not default_tagged and cfg == search_space.default_overrides:
            plan.append(("default", cfg))
            default_tagged = True
        else:
            plan.append((f"r{i}", cfg))
    return plan


def tune(
    model_cls: Type[RelArenaModel],
    search_space: SearchSpaceProvider,
    task: EntityTask,
    split: InnerSplit,
    *,
    n_trials: int,
    seed: int = 0,
    time_limit_per_trial: float | None = None,
    cache_predictions: bool = True,
    cache: CacheConfig | None = None,
    run_identity: RunIdentity | None = None,
) -> list[TrialResult]:
    """Evaluate the search space's config plan (`plan_configs`) on `split`.

    A size-aware search-space factory is resolved here, where the task's scale is
    known (from the inner-split train table), to the concrete `SearchSpace`.
    Returns the trials in execution order. `seed` makes config sampling
    reproducible.
    """
    stats = TaskStats(num_train_nodes=len(split.train_table.df))
    plan = plan_configs(resolve_search_space(search_space, stats), n_trials, seed)

    return [
        run_trial(
            model_cls,
            config,
            config_tag,
            task,
            split,
            seed=seed,
            time_limit=time_limit_per_trial,
            cache_predictions=cache_predictions,
            cache=cache,
            run_identity=run_identity,
        )
        for config_tag, config in plan
    ]
