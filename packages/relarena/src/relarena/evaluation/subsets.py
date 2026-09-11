"""Task subsets for filtering a leaderboard.

A subset is a mask over a RelArena results frame. `compute_leaderboard` and
`write_leaderboard_plots` take one as `subset=`, applied *before* their density
check: the filtered frame's task set is the one methods must cover, so a method
present on only part of the full frame can still rank on a subset it does cover.

`SUBSETS` maps names to masks — `"all"`, the default no-op, plus two examples.
Anything else is a mask of your own. Which slices make a comparable board is
the caller's call, not this module's.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

#: A subset mask: a results frame in, a boolean `Series` out.
TaskMask = Callable[[pd.DataFrame], pd.Series]

#: Example subsets, keyed on the results schema's own columns. `task_type` holds
#: `TaskType` *names* (`summary_to_dataframe` writes `task_type.name`), the same
#: spelling `plots._TASK_TYPE_LABELS` uses.
SUBSETS: dict[str, TaskMask] = {
    "all": lambda d: pd.Series(True, index=d.index),
    "classification": lambda d: d["task_type"] == "BINARY_CLASSIFICATION",
    "regression": lambda d: d["task_type"] == "REGRESSION",
}


def apply_subset(results: pd.DataFrame, subset: str | TaskMask) -> pd.DataFrame:
    """Filter a results frame to `subset`: a `SUBSETS` name or a mask of your own."""
    if isinstance(subset, str):
        if subset not in SUBSETS:
            raise ValueError(
                f"Unknown task subset {subset!r}; known: {sorted(SUBSETS)}."
            )
        subset = SUBSETS[subset]
    return results[subset(results)]
