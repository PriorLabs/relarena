"""Generate the README leaderboard tables and joint plot from checked-in results."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from relarena.evaluation import compute_leaderboard, method_kind

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN GENERATED LEADERBOARDS -->"
END = "<!-- END GENERATED LEADERBOARDS -->"
MODEL_COLOR = "#557a95"
CONSTANT_COLOR = "#b8bdc4"
INK = "#303740"
PLOT_CAPTION = (
    "Entries marked as systems (hatched) comply "
    "with RelArena-α's data states and evaluation regime but not with its standardized "
    "tuning regime. Each panel reports slightly different Elo for the same method, "
    "since Elo ratings are relative."
)
CAPTION = (
    "**Models** (solid bars) use RelArena's standardized tuning regime. "
    "**Systems** (hatched bars) follow the same data states and evaluation protocol "
    "but bring their own tuning procedure. The left panel compares models; "
    "the right includes both models and systems. Each panel computes Elo separately, "
    "so the same method can have different ratings in the two panels. "
    "See [models and systems](docs/adding-a-model.md#model-or-system) "
    "for the submission contracts."
)


def draw_board(ax: Axes, board: pd.DataFrame, title: str) -> None:
    """Draw bars from the global-constant anchor with full bootstrap intervals."""
    # Hide only the plotted row; LightGBM still contributes to Elo and tables.
    board = board.loc[board.index != "lightgbm"]
    anchor = float(board.loc["constant-global", "elo"])
    for position, (method, row) in enumerate(board.iterrows()):
        system = method_kind(method) == "system"
        color = (
            CONSTANT_COLOR
            if method in {"constant-global", "constant-per-entity"}
            else MODEL_COLOR
        )
        ax.barh(
            position,
            row["elo"] - anchor,
            left=anchor,
            height=0.62,
            color="white" if system else color,
            edgecolor=MODEL_COLOR if system else "none",
            hatch="///" if system else None,
            linewidth=1,
            zorder=2,
        )
        lower, upper = row["elo"] - row["elo-"], row["elo"] + row["elo+"]
        ax.hlines(position, lower, upper, color=INK, linewidth=1, zorder=3)
        ax.vlines(
            [lower, upper],
            position - 0.07,
            position + 0.07,
            color=INK,
            linewidth=1,
            zorder=3,
        )
        ax.text(
            upper + 18,
            position,
            f"{row['elo']:.1f}",
            va="center",
            fontsize=10,
            color=INK,
        )
    ax.set_yticks(range(len(board)), board.index)
    ax.invert_yaxis()
    ax.axvline(anchor, color="#9299a2", linewidth=0.8, zorder=1)
    ax.set_xlim(
        min(anchor, float((board["elo"] - board["elo-"]).min())) - 25,
        float((board["elo"] + board["elo+"]).max()) + 145,
    )
    ax.set_xlabel(f"Elo (anchored at {anchor:.0f})", labelpad=10)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(axis="x", color="#d7dce1", linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9299a2")


def plot_boards(boards: list[tuple[str, pd.DataFrame]]) -> bytes:
    """Render leaderboard panels with a shared legend and standard fonts."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(
            1,
            len(boards),
            figsize=(
                7 * len(boards),
                0.40 * max((b.index != "lightgbm").sum() for _, b in boards) + 2.2,
            ),
            squeeze=False,
        )
        for ax, (title, board) in zip(axes[0], boards, strict=True):
            draw_board(ax, board, title)
        handles = [
            Patch(facecolor=MODEL_COLOR, label="Model"),
            Patch(facecolor=CONSTANT_COLOR, label="Constant predictor"),
        ]
        if any(any(method_kind(m) == "system" for m in b.index) for _, b in boards):
            handles.append(
                Patch(
                    facecolor="white",
                    edgecolor=MODEL_COLOR,
                    hatch="///",
                    label="System",
                )
            )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.10),
            ncol=3,
            frameon=False,
        )
        fig.text(
            0.5,
            0.065,
            fill(PLOT_CAPTION, width=125),
            ha="center",
            fontsize=9,
            color=INK,
        )
        fig.text(
            0.5,
            0.025,
            "Higher Elo is better · 95% bootstrap intervals",
            ha="center",
            fontsize=9,
            color=INK,
        )
        fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=3)
        buffer = BytesIO()
        fig.savefig(
            buffer, format="png", dpi=180, facecolor="white", bbox_inches="tight"
        )
        plt.close(fig)
        return buffer.getvalue()


def render_board(board: pd.DataFrame, title: str) -> str:
    """Render a collapsible Markdown leaderboard table."""
    lines = [
        "<details>",
        f"<summary><b>{title} leaderboard table</b></summary>",
        "",
        "| Rank | Method | Kind | Elo | 95% bootstrap interval |",
        "|---:|---|---|---:|---:|",
    ]
    for position, (method, row) in enumerate(board.iterrows(), start=1):
        lower, upper = row["elo"] - row["elo-"], row["elo"] + row["elo+"]
        lines.append(
            f"| {position} | {method} | {method_kind(method).capitalize()} "
            f"| {row['elo']:.1f} | {lower:.1f}–{upper:.1f} |"
        )
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def main() -> int:
    """Update generated content, or check that it matches the source results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if generated content is stale"
    )
    args = parser.parse_args()
    plt.switch_backend("Agg")
    results = pd.read_csv(ROOT / "baseline_results/results.csv")
    sections = [
        "![Models and models + systems: Elo ratings with 95% bootstrap intervals](docs/leaderboards.png)",
        CAPTION,
    ]
    boards = []
    outputs: dict[Path, bytes] = {}
    for title, kinds in (
        ("Models", frozenset({"model"})),
        ("Models + systems", None),
    ):
        board = compute_leaderboard(results, kinds=kinds).sort_values(
            "elo", ascending=False, kind="stable"
        )
        boards.append((title, board))
        sections.append(render_board(board, title))
    outputs[ROOT / "docs/leaderboards.png"] = plot_boards(boards)
    readme = ROOT / "README.md"
    before, rest = readme.read_text().split(START)
    _, after = rest.split(END)
    outputs[readme] = (
        before + START + "\n\n" + "\n\n".join(sections) + "\n\n" + END + after
    ).encode()
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(path)
        else:
            path.write_bytes(content)
            print(path)
    if stale:
        for path in stale:
            print(f"Stale leaderboard content: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
