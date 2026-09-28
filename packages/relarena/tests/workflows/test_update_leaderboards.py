"""Checks for generated leaderboard content and stale-output detection."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from relarena_core.discovery import discover_models
from relarena_core.registry import registry

pytest.importorskip("matplotlib")
pytest.importorskip("bencheval.evaluator")

ROOT = Path(__file__).resolve().parents[4]


def _generate(root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "update_leaderboards", ROOT / "workflows/update_leaderboards.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (root / "baseline_results").mkdir(parents=True)
    (root / "docs").mkdir()
    shutil.copyfile(
        ROOT / "baseline_results/results.csv", root / "baseline_results/results.csv"
    )
    (root / "README.md").write_text(f"Before\n{module.START}\n{module.END}\nAfter\n")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["update_leaderboards.py"])
    assert module.main() == 0
    return module


@pytest.fixture
def generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _generate(tmp_path, monkeypatch)


def test__generate__separate_boards_and_preserve_surrounding_text(
    generated: ModuleType,
) -> None:
    text = (generated.ROOT / "README.md").read_text()
    assert text.startswith("Before\n") and text.endswith("\nAfter\n")
    assert [p.name for p in (generated.ROOT / "docs").glob("*.png")] == [
        "leaderboards.png"
    ]
    models, combined = text.split(
        "<summary><b>Models + systems leaderboard table</b></summary>"
    )
    assert "| System |" not in models
    for system in ("rt-plurel", "kurversc"):
        assert system not in models
        assert f"| {system} | System |" in combined
    assert models.count("| Model |") == combined.count("| Model |")
    assert "| lightgbm | Model |" in models
    assert "| lightgbm | Model |" in combined


def test__plot__hides_lightgbm_without_changing_ratings(generated: ModuleType) -> None:
    results = generated.pd.read_csv(generated.ROOT / "baseline_results/results.csv")
    board = generated.compute_leaderboard(results)
    original = board.copy(deep=True)
    fig, ax = generated.plt.subplots()
    try:
        generated.draw_board(ax, board, "Models + systems")
        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert labels == [method for method in board.index if method != "lightgbm"]
        assert [label.get_text() for label in ax.texts] == [
            f"{board.loc[method, 'elo']:.1f}" for method in labels
        ]
        generated.pd.testing.assert_frame_equal(board, original)
    finally:
        generated.plt.close(fig)


@pytest.mark.parametrize("stale_path", [None, "README.md", "docs/leaderboards.png"])
def test__check__detects_stale_content_without_writing(
    generated: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    stale_path: str | None,
) -> None:
    if stale_path is not None:
        path = generated.ROOT / stale_path
        path.write_bytes(
            path.read_bytes().replace(b"tabpfn-rel-client-2026-08-15", b"stale") + b"\n"
        )
    paths = [
        generated.ROOT / "README.md",
        *sorted((generated.ROOT / "docs").glob("*.png")),
    ]
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(sys, "argv", ["update_leaderboards.py", "--check"])
    assert generated.main() == (0 if stale_path is None else 1)
    assert before == {path: path.read_bytes() for path in paths}


def test__generate__moving_an_alias__only_relabels_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discover_models()
    plain = (_generate(tmp_path / "plain", monkeypatch).ROOT / "README.md").read_text()
    for target, other in (("relgt", "relgnn-es"), ("relgnn-es", "relgt")):
        monkeypatch.setattr(registry, "_aliases", {})
        registry.register_alias("graph-latest", target)
        root = _generate(tmp_path / target, monkeypatch).ROOT
        text = (root / "README.md").read_text()

        assert f"| {target} |" not in text
        assert "| graph-latest | Model |" in text
        assert f"| {other} | Model |" in text
        assert text.replace("| graph-latest |", f"| {target} |") == plain
        results = "baseline_results/results.csv"
        assert (root / results).read_bytes() == (ROOT / results).read_bytes()
