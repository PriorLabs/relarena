"""Adapt a fitted sklearn-style estimator's output to the evaluate contract.

Any baseline that wraps an sklearn-style estimator (one exposing `predict` and, for
classifiers, `predict_proba` + `classes_`) has to reshape that estimator's output
into what `EntityTask.evaluate` expects — and that reshaping is identical regardless
of *which* estimator it is. So it lives here, shared by the `constant-global`
baseline and the `rdblearn` TFM core.

The contract:

  * regression  -> `(N,)`
  * binary      -> `(N,)`, the probability of the positive class (`0.0` if the
    positive class never appeared in training)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from relbench.base import TaskType


def predict_to_contract(estimator: Any, X: Any, task_type: TaskType) -> np.ndarray:
    """Predict with `estimator` on `X` and shape it to the evaluate contract.

    Raises on any unsupported task type.
    """
    if task_type == TaskType.REGRESSION:
        return np.asarray(estimator.predict(X), dtype=float)

    if task_type == TaskType.BINARY_CLASSIFICATION:
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
        classes_list = np.asarray(estimator.classes_).tolist()
        if 1 in classes_list:
            return proba[:, classes_list.index(1)]
        return np.zeros(proba.shape[0], dtype=float)

    raise ValueError(f"Unsupported task type for prediction: {task_type}.")
