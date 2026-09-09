"""Cross-experiment reporting: leaderboards, plots, and reference baselines.

Everything here consumes a finished results frame and never takes part in a run,
so nothing in the harness core (`model`, `tuner`, `runner`, `metrics`) may import
it. `reference` supplies externally-reported scores to rank alongside our own,
and `subsets` names the task slices a custom leaderboard can be built on.
"""

from relarena.evaluation.leaderboard import (
    compute_leaderboard,
    method_kind,
    to_bencheval_frame,
)
from relarena.evaluation.plots import (
    plot_critical_difference,
    plot_normalized_loss_heatmap,
    write_leaderboard_plots,
)
from relarena.evaluation.reference import load_reference_results
from relarena.evaluation.subsets import SUBSETS, TaskMask, apply_subset

__all__ = [
    "SUBSETS",
    "TaskMask",
    "apply_subset",
    "compute_leaderboard",
    "load_reference_results",
    "method_kind",
    "plot_critical_difference",
    "plot_normalized_loss_heatmap",
    "to_bencheval_frame",
    "write_leaderboard_plots",
]
