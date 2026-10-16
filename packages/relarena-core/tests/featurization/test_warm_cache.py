"""Tests for the directly runnable shared DFS warmer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from relarena_core.cache import CacheConfig
from relarena_core.featurization import warm_cache
from relarena_core.identity import RunIdentity

INNER = SimpleNamespace(db_state="inner-db", train_table="it", eval_table="iv")
OUTER = SimpleNamespace(
    db_state="outer-db", train_table="ot", val_table="ov", eval_table="oe"
)


def _fake_source(
    monkeypatch: pytest.MonkeyPatch,
    inner_split: Callable[[], SimpleNamespace] = lambda: INNER,
    outer_split: Callable[[], SimpleNamespace] = lambda: OUTER,
) -> tuple[SimpleNamespace, list[tuple[tuple, dict]]]:
    """A fake task source and the recorded `build_dfs_features` calls."""
    source = SimpleNamespace(
        task=SimpleNamespace(time_col="time"),
        inner_split=inner_split,
        outer_split=outer_split,
        run_identity=lambda phase, **kwargs: RunIdentity(
            "dataset", "db", "task", "labels", phase=phase
        ),
    )
    monkeypatch.setattr(warm_cache, "concat_tables", lambda a, b: "ot+ov")
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        warm_cache,
        "build_dfs_features",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return source, calls


def _unneeded() -> SimpleNamespace:
    raise AssertionError("a split no selected group needs was built")


def test__warm_dfs_cache__uses_one_shared_preprocessor_for_both_phases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, calls = _fake_source(monkeypatch)
    config = CacheConfig(tmp_path, "fill")

    warm_cache.warm_dfs_cache(source, config, max_depth=3)

    assert [(args[1], args[2], kwargs["history_table"]) for args, kwargs in calls] == [
        ("inner-db", "it", "it"),
        ("inner-db", "iv", "it"),
        ("outer-db", "ot", "ot"),
        ("outer-db", "oe", "ot"),
        ("outer-db", "ot+ov", "ot+ov"),
        ("outer-db", "oe", "ot+ov"),
    ]
    assert {kwargs["run_identity"].phase for _, kwargs in calls} == {
        "inner",
        "outer",
    }
    assert all(kwargs["cache"] is config for _, kwargs in calls)
    assert all(kwargs["max_depth"] == 3 for _, kwargs in calls)


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("inner", [("inner-db", "it", "it"), ("inner-db", "iv", "it")]),
        ("outer-train", [("outer-db", "ot", "ot"), ("outer-db", "oe", "ot")]),
        ("outer-full", [("outer-db", "ot+ov", "ot+ov"), ("outer-db", "oe", "ot+ov")]),
    ],
)
def test__warm_dfs_cache__one_group__warms_only_its_matrices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    group: str,
    expected: list[tuple[str, str, str]],
) -> None:
    if group == "inner":
        source, calls = _fake_source(monkeypatch, outer_split=_unneeded)
    else:
        source, calls = _fake_source(monkeypatch, inner_split=_unneeded)

    warm_cache.warm_dfs_cache(source, CacheConfig(tmp_path, "fill"), groups=[group])

    assert [
        (args[1], args[2], kwargs["history_table"]) for args, kwargs in calls
    ] == expected


def test__warm_dfs_cache__unknown_group__raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown DFS history groups"):
        warm_cache.warm_dfs_cache(None, CacheConfig(tmp_path, "fill"), groups=["outer"])


def test__warm_dfs_cache__non_fill_config__raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="on_miss='fill'"):
        warm_cache.warm_dfs_cache(None, CacheConfig(tmp_path, "raise"))
