"""Local experiment runner.

Ties the pieces together for a single `(model, dataset, task)`: load the
RelBench task, check the model supports the task type, run the tuner, and select
the default / tuned configurations. Runs locally and in-process — cluster
orchestration is out of scope.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type

import numpy as np
from relbench.base import TaskType

from relarena.cache import resolve_cache_config
from relarena.dataset import RelBenchDatasetTask
from relarena.metrics import is_better
from relarena.model import RelArenaModel
from relarena.registry import registry
from relarena.results import SystemResult, TrialResult
from relarena.search_space import SearchSpaceProvider
from relarena.system import RelArenaSystem
from relarena.tasks import ENTITY_TASK_TYPES
from relarena.tuner import refit_and_evaluate, tune

logger = logging.getLogger(__name__)


def select_best(trials: list[TrialResult], metric: Callable[..., float]) -> TrialResult:
    """Pick the trial with the best validation score under `metric`'s direction."""
    valid = [
        t
        for t in trials
        if t.ok and t.val_score is not None and math.isfinite(t.val_score)
    ]
    if not valid:
        raise RuntimeError(
            "No successful trials with a finite validation score to select from."
        )
    best = valid[0]
    for t in valid[1:]:
        if is_better(t.val_score, best.val_score, metric):
            best = t
    return best


@dataclass
class ExperimentSummary:
    """The headline result for one `(model, dataset, task, seed)` run."""

    model_name: str
    dataset: str
    task_name: str
    task_type: TaskType
    metric_name: str
    seed: int
    n_trials: int  # the requested HPO budget
    default: TrialResult | None  # the zero-tuning config
    tuned: TrialResult | None  # best config by validation score
    trials: list[TrialResult]  # all trials and their analysis metadata
    kind: str = "model"


@dataclass
class SystemExperimentSummary:
    """The result for one end-to-end `(system, dataset, task, seed)` run."""

    system_name: str
    dataset: str
    task_name: str
    task_type: TaskType
    metric_name: str
    seed: int
    result: SystemResult


def run_system_experiment(
    system_cls: Type[RelArenaSystem],
    dataset_name: str,
    task_name: str,
    *,
    seed: int = 0,
    time_limit: float | None = None,
    download: bool = True,
    cache_predictions: bool = True,
    cache_dir: str | Path | None = None,
    evaluate_test: bool = True,
) -> SystemExperimentSummary:
    """Run one system with both protocol splits and record its final result.

    RelArena constructs the correctly censored `InnerSplit` and `OuterSplit`
    objects, withholds test labels, and evaluates the returned predictions. The
    system owns every step between receiving the splits and returning those
    predictions; in particular, it may use the inner split for selection or
    ignore it.
    """
    source = RelBenchDatasetTask(dataset_name, task_name, download=download)
    task = source.task
    cache = resolve_cache_config(cache_dir, on_miss="raise")

    assert task.task_type in ENTITY_TASK_TYPES, (
        f"RelArena only handles regression and binary classification tasks; got "
        f"{task.task_type} for task '{task_name}'."
    )
    if task.task_type not in system_cls.supported_task_types:
        raise ValueError(
            f"System '{system_cls.name}' does not support task_type "
            f"{task.task_type} (task '{task_name}')."
        )

    inner = source.inner_split()
    outer = source.outer_split()
    system = system_cls(cache=cache, run_identity=source.run_identity())
    started = time.perf_counter()
    predictions = np.asarray(
        system.run(
            task,
            inner_split=inner,
            outer_split=outer,
            seed=seed,
            time_limit=time_limit,
        )
    )
    time_total = time.perf_counter() - started
    expected_shape = (len(outer.eval_table.df),)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"System '{system_cls.name}' returned predictions with shape "
            f"{predictions.shape}; expected {expected_shape}."
        )

    metric = source.metric
    test_metrics: dict[str, float] = {}
    test_score = None
    if evaluate_test:
        metrics = list(task.metrics)
        if metric.__name__ not in {m.__name__ for m in metrics}:
            metrics.append(metric)
        test_metrics = task.evaluate(predictions, None, metrics=metrics)
        test_score = float(test_metrics[metric.__name__])

    return SystemExperimentSummary(
        system_name=system_cls.name,
        dataset=dataset_name,
        task_name=task_name,
        task_type=task.task_type,
        metric_name=metric.__name__,
        seed=seed,
        result=SystemResult(
            test_score=test_score,
            test_metrics=test_metrics,
            time_total=time_total,
            test_pred=predictions if cache_predictions else None,
        ),
    )


def run_model_experiment(
    model_cls: Type[RelArenaModel],
    dataset_name: str,
    task_name: str,
    *,
    search_space: SearchSpaceProvider | None = None,
    seed: int = 0,
    n_trials: int = 10,
    time_limit_per_trial: float | None = None,
    download: bool = True,
    cache_predictions: bool = True,
    cache_dir: str | Path | None = None,
    evaluate_test: bool = True,
    require_all_trials: bool = True,
) -> ExperimentSummary:
    """Tune one model on a RelBench entity task and summarize its trials.

    `search_space` defaults to the one registered for `model_cls` (via
    `@register_model`); pass it explicitly to override.

    Protocol (nested temporal validation; see docs/temporal-validation.md):
      1. **Tune** — fit each config on `train`, score on `val`, using the DB
         censored at `val_timestamp` so validation features are frozen at the val
         cutoff (mirroring how test features are frozen at the test cutoff). No
         per-config test prediction is made.
      2. **Select** — pick the best config by validation score. Every trial must
         have succeeded; one failure raises rather than selecting from a partial
         grid, unless `require_all_trials` is off, which selects from whichever
         trials survived (the score is then not comparable to a full run).
      3. **Final fit** — fit the selected config(s) under each model's declared
         final-fit regime and produce the test prediction, using the database
         censored at `test_timestamp`.

    The task type must be in `model_cls.supported_task_types` (entity tasks
    only). Returns an `ExperimentSummary` with the default and best-tuned
    trials (the latter carrying the refit test score) plus the full trial list.
    """
    source = RelBenchDatasetTask(dataset_name, task_name, download=download)
    task = source.task
    cache = resolve_cache_config(cache_dir, on_miss="raise")

    # Guard the supported-task-type invariant in the live path, independent of any
    # per-model `supported_task_types` override.
    assert task.task_type in ENTITY_TASK_TYPES, (
        f"RelArena only handles regression and binary classification tasks; got "
        f"{task.task_type} for task '{task_name}'."
    )

    if task.task_type not in model_cls.supported_task_types:
        raise ValueError(
            f"Model '{model_cls.name}' does not support task_type "
            f"{task.task_type} (task '{task_name}')."
        )

    if search_space is None:
        search_space = registry.search_space_for(model_cls)

    metric = source.metric

    # Phase 1+2: tune on the inner split (train→val, DB censored at val_timestamp so
    # validation features are frozen at the val cutoff — see the protocol note above
    # and docs/temporal-validation.md), then select the best config by val score.
    trials = tune(
        model_cls,
        search_space,
        task,
        source.inner_split(),
        n_trials=n_trials,
        seed=seed,
        time_limit_per_trial=time_limit_per_trial,
        cache_predictions=cache_predictions,
        cache=cache,
        run_identity=source.run_identity("inner"),
    )

    # Selecting from a partial grid silently changes the protocol: the reported
    # test score is then the best of however many configs happened to survive,
    # which is not comparable to a run that evaluated the full grid. A transient
    # failure must fail the experiment so it can be re-run, not shrink it.
    failed = [t for t in trials if not t.ok]
    if require_all_trials and failed:
        raise RuntimeError(
            f"{len(failed)} of {len(trials)} trials failed for '{model_cls.name}' "
            f"on '{dataset_name}/{task_name}'; refusing to select and refit from "
            f"a partial grid. First failure (config_id={failed[0].config_id}): "
            f"{failed[0].error}"
        )

    default = next((t for t in trials if t.config_tag == "default"), None)
    tuned = select_best(trials, metric) if any(t.ok for t in trials) else None

    # Phase 3: run the selected config(s) under the model's final-fit regime on the
    # outer split (DB censored at test_timestamp) to get the test score.
    if evaluate_test and tuned is not None:
        outer = source.outer_split()
        to_refit = [tuned]
        if default is not None and default.ok and default.config_id != tuned.config_id:
            to_refit.append(default)
        for trial in to_refit:
            try:
                refit = refit_and_evaluate(
                    model_cls,
                    trial.config,
                    task,
                    outer,
                    seed=seed,
                    time_limit=time_limit_per_trial,
                    cache=cache,
                    run_identity=source.run_identity("outer"),
                )
                trial.test_score = refit["test_score"]
                trial.test_metrics = refit["test_metrics"]
                trial.test_pred = refit["test_pred"] if cache_predictions else None
                trial.fit_time_refit = refit["fit_time_refit"]
                trial.predict_time_refit = refit["predict_time_refit"]
            except Exception:
                logger.exception(
                    "Refit failed (model=%s, dataset=%s, task=%s, config_id=%s)",
                    model_cls.name,
                    dataset_name,
                    task_name,
                    trial.config_id,
                )

    return ExperimentSummary(
        model_name=model_cls.name,
        dataset=dataset_name,
        task_name=task_name,
        task_type=task.task_type,
        metric_name=metric.__name__,
        seed=seed,
        n_trials=n_trials,
        default=default,
        tuned=tuned,
        trials=trials,
        kind=getattr(model_cls, "kind", "model"),
    )


def run_experiment(
    method_cls: Type[RelArenaModel] | Type[RelArenaSystem],
    dataset_name: str,
    task_name: str,
    *,
    search_space: SearchSpaceProvider | None = None,
    seed: int = 0,
    n_trials: int = 10,
    time_limit_per_trial: float | None = None,
    download: bool = True,
    cache_predictions: bool = True,
    cache_dir: str | Path | None = None,
    evaluate_test: bool = True,
    require_all_trials: bool = True,
) -> ExperimentSummary | SystemExperimentSummary:
    """Dispatch one experiment to the model or system runner.

    Call `run_model_experiment` or `run_system_experiment` directly when the
    method kind is already known. This wrapper keeps the CLI and registry-based
    callers uniform. Model-only search arguments do not affect a native system;
    `time_limit_per_trial` becomes its single soft time limit.
    """
    if isinstance(method_cls, type) and issubclass(method_cls, RelArenaSystem):
        if search_space is not None:
            raise TypeError("A RelArenaSystem does not have a harness search space.")
        return run_system_experiment(
            method_cls,
            dataset_name,
            task_name,
            seed=seed,
            time_limit=time_limit_per_trial,
            download=download,
            cache_predictions=cache_predictions,
            cache_dir=cache_dir,
            evaluate_test=evaluate_test,
        )

    return run_model_experiment(
        method_cls,
        dataset_name,
        task_name,
        search_space=search_space,
        seed=seed,
        n_trials=n_trials,
        time_limit_per_trial=time_limit_per_trial,
        download=download,
        cache_predictions=cache_predictions,
        cache_dir=cache_dir,
        evaluate_test=evaluate_test,
        require_all_trials=require_all_trials,
    )
