"""Load externally-reported reference baselines for the leaderboard.

Some methods on RelBenchV1 we do not run ourselves -- closed-source relational
foundation models and fully-supervised GNNs -- but their per-task scores are
published. `reference_results.csv` is a curated snapshot of those numbers.

They are reference points only -- mostly self-reported by the methods' authors
(some recomputed by us, but outside `relarena`), and a few follow a different
evaluation protocol that can overestimate performance. Their method names carry a
`_MR` (model report) suffix to keep that caveat visible on every board.

`load_reference_results` returns the snapshot in the RelArena results shape so
it flows through the same `relarena.evaluation.to_bencheval_frame`
adapter as real results -- pass it as the `reference=` argument of
`relarena.evaluation.compute_leaderboard` or
`relarena.evaluation.write_leaderboard_plots` to include the chosen methods as
ordinary baselines. The reference rows carry no tuning sweep, no validation
scores, and no timing: only the reported test score in the task's primary metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Columns the bencheval adapter reads but the curated CSV omits (reported
#: numbers have no tuning sweep or timing). Added on load -- as NaN, since the
#: values are genuinely unknown -- so a reference frame drops straight into
#: `to_bencheval_frame` (which coerces the NaN times to 0).
_ADAPTER_TIME_COLUMNS = ("fit_time_tuning", "fit_time_refit", "predict_time_refit")


def load_reference_results(
    path: str | Path,
    *,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Load reference baselines as a RelArena-shaped results frame.

    Reads the curated reference CSV (`source, model, dataset, task, task_type,
    metric, test_score`) and adds the columns the leaderboard adapter expects
    but the file omits: `selected=True` and NaN timing columns.

    Args:
        path: Path to the reference results CSV.
        methods: Keep only these reference method names; raises if any requested
            method is absent. `None` (default) keeps every method in the file.
    """
    frame = pd.read_csv(path)
    if methods is not None:
        available = set(frame["model"])
        missing = sorted(set(methods) - available)
        if missing:
            raise ValueError(
                f"Reference methods not found in {path}: {missing}. "
                f"Available: {sorted(available)}"
            )
        frame = frame[frame["model"].isin(methods)].reset_index(drop=True)
    frame["selected"] = True
    for col in _ADAPTER_TIME_COLUMNS:
        frame[col] = np.nan
    # Mirror the reported score into its native `test_<metric>` column, like real
    # results carry both, so reference methods also appear in the raw-metric
    # heatmaps for the metric they report (the report has no other native metric,
    # so e.g. regression references stay out of the raw R² heatmap).
    for metric_name in frame["metric"].unique():
        is_metric = frame["metric"] == metric_name
        frame.loc[is_metric, f"test_{metric_name}"] = frame.loc[is_metric, "test_score"]
    return frame
