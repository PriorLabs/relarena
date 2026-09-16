"""Validation-based configuration selection."""

import math
from typing import Callable

from relarena.core.metrics import is_better
from relarena.core.results import TrialResult


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
