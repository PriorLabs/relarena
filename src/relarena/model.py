"""The model contract.

Every model that RelArena can run implements `RelArenaModel`. The split of
responsibilities is deliberate (and is the core fairness lever):

  * the **model** owns the training math; its hyperparameter *search space* is a
    separate `SearchSpace`, bound to the model in the
    registry (not declared on the class — mirroring AutoGluon / TabArena);
  * the **harness** owns the training loop's invocation, the HPO *budget*
    (number of trials / wall-clock), the splits, and the evaluation protocol.

A model *instance* corresponds to a single hyperparameter configuration; the
tuner instantiates one model per config drawn from the search space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from relbench.base import Database, EntityTask, Table, TaskType

from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.tasks import ENTITY_TASK_TYPES


class RelArenaModel(ABC):
    """Base contract for an entity-task model.

    Subclasses set `name`, optionally narrow `supported_task_types`,
    and implement `fit` and `predict`. The hyperparameter search space
    is **not** part of this contract — it is a separate
    `SearchSpace` registered alongside the model via
    `@register_model(search_space=...)`.
    """

    #: Unique, human-readable identifier used in the registry and result tables.
    name: ClassVar[str]

    #: Task types this model can handle. Defaults to all entity task types; the
    #: runner refuses to run a model on a task whose type is not in this set.
    supported_task_types: ClassVar[frozenset[TaskType]] = ENTITY_TASK_TYPES

    #: How the selected config is fit for the final test prediction. `True`
    #: (default): refit on the train+val union and report that model. `False`:
    #: train on `train` alone, pass `val` for early stopping / checkpoint
    #: selection, and report the selected checkpoint — for methods whose published
    #: protocol reports a best-val checkpoint rather than a train+val refit.
    refit_on_full_data: ClassVar[bool] = True

    #: `"model"` (selection through the registered search space) or `"system"`
    #: (selection inside `fit`). Leaderboards and plots read this to keep the
    #: two populations separable — they are not the same kind of result. What
    #: qualifies, the experimental status of system support, and the submission
    #: rules live in docs/adding-a-model.md, "Model or system?".
    kind: ClassVar[str] = "model"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        cache: CacheConfig | None = None,
        run_identity: RunIdentity | None = None,
    ) -> None:
        """Instantiate the model for a single hyperparameter `config`.

        `config` is one config produced by the model's registered search space
        (the default regime passes the search space's `default_overrides`);
        `fit` / `predict` read it via `self.config`.
        """
        #: The resolved hyperparameter configuration for this instance.
        self.config: dict[str, Any] = dict(config)
        self.cache = cache or CacheConfig(directory=None, on_miss="compute")
        self.run_identity = run_identity

    # -- train / predict (the math; the harness drives the loop) -------------

    @abstractmethod
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
        """Train on `train_table`.

        `val_table` may be used for early stopping / model selection internal to
        the model. It is `None` when refitting the selected config on train+val
        for the final test prediction — there is no held-out split then, so a model
        with early stopping should fall back to a fixed budget (e.g. the iteration
        count found during tuning). `db` is the (test-time-censored) relational
        database the model may materialize features/graphs from. `time_limit` is a
        soft wall-clock budget in seconds (`None` = unbounded).
        """

    @abstractmethod
    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Predict for the entities in `table`.

        Returns predictions in the shape `EntityTask.evaluate` expects:
        `(len(table),)`.
        """
