"""Configuration planning and trial error reporting."""

import logging

import pytest
from ConfigSpace import ConfigurationSpace, Integer

from relarena_core.search_space import SearchSpace
from relarena_core.tuner import _concise_error, plan_configs


def _random_space() -> SearchSpace:
    return SearchSpace(
        space=ConfigurationSpace(space=[Integer("x", (1, 100))], seed=0),
        default_overrides={},
    )


def _grid_space() -> SearchSpace:
    return SearchSpace(
        fixed_grid=[{"d": 3}, {"d": 2}, {"d": 1}], default_overrides={"d": 2}
    )


def test_plan_configs_random_default_plus_samples() -> None:
    plan = plan_configs(_random_space(), n_trials=3, seed=0)
    tags = [t for t, _ in plan]
    assert tags[0] == "default" and plan[0][1] == {}  # the empty default comes first
    assert len(plan) == 4  # default + 3 random samples
    assert all("x" in cfg for _, cfg in plan[1:])


def test_plan_configs_grid_uses_grid_in_order() -> None:
    plan = plan_configs(_grid_space(), n_trials=99, seed=0)
    configs = [c for _, c in plan]
    assert configs == [{"d": 3}, {"d": 2}, {"d": 1}]  # whole grid, deepest first
    assert plan[1] == ("default", {"d": 2})  # default-matching entry tagged "default"


def test_plan_configs_grid_capped_at_n_trials_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        plan = plan_configs(_grid_space(), n_trials=2, seed=0)
    configs = [c for _, c in plan]
    # budget < grid -> keep the first n_trials (the deepest-first grid keeps d=3, d=2)
    assert configs == [{"d": 3}, {"d": 2}]
    # default still tagged when it survives the cap
    assert plan[1] == ("default", {"d": 2})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("dropping 1" in m for m in warnings)


def test_plan_configs_grid_within_budget_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        plan_configs(_grid_space(), n_trials=3, seed=0)  # exactly fits
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_default_overrides_not_in_fixed_grid_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        SearchSpace(fixed_grid=[{"d": 3}, {"d": 1}], default_overrides={"d": 2})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not in the fixed_grid" in m for m in warnings)


def test_default_overrides_in_fixed_grid_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        SearchSpace(fixed_grid=[{"d": 3}, {"d": 2}], default_overrides={"d": 2})
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test__concise_error__one_line_type_message_and_raise_site() -> None:
    try:
        raise ValueError("bad\nstuff")  # multi-line message must collapse to one line
    except ValueError as exc:
        summary = _concise_error(exc)
    assert "\n" not in summary
    assert summary.startswith("ValueError: bad stuff (")
    assert "test_tuner.py:" in summary  # innermost frame = where it was raised
