"""Leaderboard plots from a RelArena results frame.

Two views, both fed by `leaderboard.to_bencheval_frame`:

- `plot_normalized_loss_heatmap` — a method × task heatmap of per-task
  normalized loss (`(err - best) / (worst - best)` in `[0, 1]`, 0 = best),
  split by task type so classification and regression read separately.
- `plot_raw_score_heatmap` — a method × task heatmap of a raw metric value
  (e.g. ROC AUC, R²), for reading absolute performance.
- `plot_winrate_matrix` — bencheval's pairwise win-rate matrix.
- `plot_critical_difference` — bencheval's critical-difference diagram (the
  Demšar significance test over mean ranks).

Plotting deps (matplotlib/seaborn, `bencheval`, `autorank`) are the
`plots` extra; they are imported lazily inside the functions so importing
this module stays dependency-free.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from relarena.evaluation.leaderboard import (
    _drop_incomplete_tasks,
    to_bencheval_frame,
)

_TASK_TYPE_LABELS = {
    "BINARY_CLASSIFICATION": "classification",
    "REGRESSION": "regression",
}


#: Dark end of the white->blue heatmap ramps.
_HEATMAP_BLUE = "#08519c"


def _normalized_loss(results: pd.DataFrame) -> pd.DataFrame:
    """Per-(method, task) `loss_rescaled` (from bencheval) + `task_type` + score.

    `loss_rescaled` is bencheval's own per-task min-max of `metric_error`
    (0 = best model on that task, 1 = worst) — *the same value the leaderboard
    ranks on*, so the heatmap and the board agree by construction. `test_score`
    is the raw primary metric value (annotated on the heatmap).
    """
    from bencheval.evaluator import BenchmarkEvaluator

    frame = BenchmarkEvaluator().compute_results_per_task(to_bencheval_frame(results))

    meta = results
    if "selected" in meta.columns:
        meta = meta[meta["selected"]]
    meta = meta[meta["test_score"].notna()].assign(
        task=lambda d: d["dataset"].astype(str) + "/" + d["task"].astype(str),
        method=lambda d: d["model"],
    )
    meta = meta[["method", "task", "task_type", "test_score"]].drop_duplicates(
        ["method", "task"]
    )
    return frame.merge(meta, on=["method", "task"], how="left")


def plot_normalized_loss_heatmap(
    results: pd.DataFrame, task_type: str, save_path: str | Path
) -> bool:
    """Write a method × task normalized-loss heatmap for one `task_type`.

    Returns `False` (writing nothing) when no task of that type is present.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    frame = _normalized_loss(results)
    frame = frame[frame["task_type"] == task_type]
    if frame.empty:
        return False

    pivot = frame.pivot(index="method", columns="task", values="loss_rescaled")
    raw = frame.pivot(index="method", columns="task", values="test_score")
    order = pivot.mean(axis=1).sort_values().index  # best methods on top
    pivot = pivot.loc[order]
    raw = raw.reindex(index=pivot.index, columns=pivot.columns)

    # Each cell: normalized loss in bold, the raw score beneath it in brackets.
    annot = pivot.copy().astype(object)
    for method in pivot.index:
        for task in pivot.columns:
            norm = pivot.loc[method, task]
            score = raw.loc[method, task]
            annot.loc[method, task] = (
                "" if pd.isna(norm) else f"$\\mathbf{{{norm:.2f}}}$\n({score:.3g})"
            )

    cmap = LinearSegmentedColormap.from_list("loss", ["#ffffff", _HEATMAP_BLUE])
    # ~0.7in per row plus generous headroom for the (rotated) task labels, so the
    # two-line cells and method names never overlap.
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.95 * pivot.shape[1]), 0.7 * pivot.shape[0] + 3.2)
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        annot=annot.to_numpy(),
        fmt="",
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "normalized loss (0 = best per task)"},
    )
    label = _TASK_TYPE_LABELS.get(task_type, task_type.lower())
    ax.set_title(f"Normalized loss per task — {label}", fontsize=13, pad=28)
    ax.text(
        0.5,
        1.015,
        "cell: normalized loss (bold, 0 = best) · raw score in (brackets)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#444444",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_raw_score_heatmap(
    results: pd.DataFrame,
    metric: str,
    save_path: str | Path,
    *,
    higher_is_better: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> bool:
    """Write a method × task heatmap of the raw `test_<metric>` value.

    `metric` is a native metric name as it appears in the results columns, e.g.
    `"roc_auc"` or `"r2"`. The cell is the val-selected config's
    `test_<metric>` per `(method, task)`; only tasks that report the metric are
    shown (so `"roc_auc"` gives the classification tasks, `"r2"` the regression
    ones). Darker = higher value; methods are ordered best-first by
    `higher_is_better`. Pass `vmin`/`vmax` to fix the colour scale (e.g.
    `vmin=0.5, vmax=1` for ROC AUC). Returns `False` if nothing reports it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    column = f"test_{metric}"
    rows = results
    if "selected" in rows.columns:
        rows = rows[rows["selected"]]
    if column not in rows.columns:
        return False
    rows = rows[rows[column].notna()]
    if rows.empty:
        return False

    rows = rows.assign(
        task=lambda d: d["dataset"].astype(str) + "/" + d["task"].astype(str),
        method=lambda d: d["model"],
    )
    pivot = rows.pivot_table(
        index="method", columns="task", values=column, aggfunc="first"
    )
    order = pivot.mean(axis=1).sort_values(ascending=not higher_is_better).index
    pivot = pivot.loc[order]  # best methods on top

    cmap = LinearSegmentedColormap.from_list("score", ["#ffffff", _HEATMAP_BLUE])
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.95 * pivot.shape[1]), 0.6 * pivot.shape[0] + 3.0)
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".3g",
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric},
    )
    ax.set_title(title or f"{metric} per task (raw)", fontsize=13, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_critical_difference(results: pd.DataFrame, save_path: str | Path) -> bool:
    """Write bencheval's critical-difference diagram over mean ranks.

    Returns `False` (writing nothing) when no dense task remains to rank.
    """
    import matplotlib.pyplot as plt
    from bencheval.evaluator import BenchmarkEvaluator

    frame = _drop_incomplete_tasks(to_bencheval_frame(results))
    if frame.empty:
        return False

    board = BenchmarkEvaluator()
    results_per_task = board.compute_results_per_task(data=frame)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # bencheval's plot_critical_diagrams does its own savefig (no dpi arg), so set
    # the resolution via rc context to match the heatmaps.
    with plt.rc_context({"savefig.dpi": 200}):
        board.plot_critical_diagrams(results_per_task, save_path=str(save_path))
    return True


def plot_winrate_matrix(results: pd.DataFrame, save_path: str | Path) -> bool:
    """Write bencheval's pairwise win-rate matrix (% of tasks method i beats j).

    Returns `False` (writing nothing) when no dense task remains to rank.
    """
    import matplotlib.pyplot as plt
    from bencheval.evaluator import BenchmarkEvaluator

    frame = _drop_incomplete_tasks(to_bencheval_frame(results))
    if frame.empty:
        return False

    board = BenchmarkEvaluator()
    results_per_task = board.compute_results_per_task(data=frame)
    matrix = board.compute_winrate_matrix(results_per_task)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # plot_winrate_matrix does its own savefig (no dpi arg); set it via rc context.
    with plt.rc_context({"savefig.dpi": 200}):
        board.plot_winrate_matrix(matrix, str(save_path), title="Pairwise win-rate (%)")
    return True


def write_leaderboard_plots(
    results: pd.DataFrame,
    out_dir: str | Path,
    *,
    reference: pd.DataFrame | None = None,
) -> list[Path]:
    """Write all leaderboard plots into `out_dir`; return the paths written.

    Two normalized-loss heatmaps (classification / regression), the pairwise
    win-rate matrix, and the critical-difference diagram. Plots with no
    applicable data are skipped.

    Pass `reference` (a frame from `relarena.evaluation.load_reference_results`)
    to show self-reported baselines alongside the reproduced runs as ordinary
    methods.
    """
    if reference is not None:
        results = pd.concat([results, reference], ignore_index=True)
    out_dir = Path(out_dir)
    written: list[Path] = []
    for task_type, label in _TASK_TYPE_LABELS.items():
        path = out_dir / f"heatmap_{label}.png"
        if plot_normalized_loss_heatmap(results, task_type, path):
            written.append(path)
    raw_specs = (
        ("roc_auc", {"vmin": 0.5, "vmax": 1.0, "title": "ROC AUC per task (raw)"}),
        ("r2", {"vmax": 1.0, "title": "R² per task (raw)"}),
    )
    for metric, kwargs in raw_specs:
        path = out_dir / f"heatmap_{metric}.png"
        if plot_raw_score_heatmap(results, metric, path, **kwargs):
            written.append(path)
    wr_path = out_dir / "winrate_matrix.png"
    if plot_winrate_matrix(results, wr_path):
        written.append(wr_path)
    cd_path = out_dir / "critical_difference.png"
    if plot_critical_difference(results, cd_path):
        written.append(cd_path)
    return written
