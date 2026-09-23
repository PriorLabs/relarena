"""The typed NoriDFS recipe and the fit-time decisions it drives."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table

from relarena.models.nori_rel import model as model_module
from relarena.models.nori_rel.recipe import DENSE_HISTORY_ROWS_PER_ENTITY, NoriDFSRecipe


def test_fixed_config_disables_every_heuristic() -> None:
    recipe = NoriDFSRecipe.from_config(model_module.NORIDFS_FIXED_CONFIG)

    assert recipe == NoriDFSRecipe()
    assert recipe.history_lags(100.0) == 0


def test_tuned_arms_share_the_heuristics_and_differ_in_budget() -> None:
    lean = NoriDFSRecipe.from_config(model_module.NORIDFS_LEAN_CONFIG)
    direct = NoriDFSRecipe.from_config(model_module.NORIDFS_DIRECT_CONFIG)
    missing = NoriDFSRecipe.from_config(model_module.NORIDFS_MISSING_CONFIG)

    for recipe in (lean, direct, missing):
        assert recipe.uses_history_lags
        assert recipe.dense_history
        assert recipe.target_lattice
        assert recipe.skew_target_transform
        assert recipe.nonnegative_support
        assert recipe.reservoir == "temporal_mix_v1"
        assert recipe.memory_policy is not None
    assert (lean.max_features, lean.max_direct_features) == (48, 1)
    assert (direct.max_features, direct.max_direct_features) == (87, 16)
    assert direct.max_direct_categorical_cardinality == 32
    assert (missing.max_features, missing.max_direct_features) == (87, 1)
    assert missing.context_profile == "v1"
    assert missing.missing_category_modes and missing.missing_fractions


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"noridfs_windows": "1,x"}, "comma-separated integers"),
        ({"noridfs_windows": "0,4"}, "positive integers"),
        ({"noridfs_windows": "4,4"}, "duplicates"),
        ({"noridfs_max_features": 0}, "budgets must be positive"),
        ({"train_profile": "v2"}, "unknown train_profile"),
        ({"noridfs_train_reservoir": "latest"}, "unknown noridfs_train_reservoir"),
        ({"noridfs_context_profile": "v9"}, "unknown noridfs_context_profile"),
    ],
)
def test_recipe_rejects_invalid_values(config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NoriDFSRecipe.from_config(config)


@pytest.mark.parametrize(("rows_per_entity", "expected_lags"), [(31, 5), (32, 1)])
def test_history_lags_use_the_dense_training_boundary(
    rows_per_entity: int, expected_lags: int
) -> None:
    task = SimpleNamespace(entity_col="entity")
    train = Table(
        pd.DataFrame({"entity": np.repeat([1, 2], rows_per_entity)}),
        fkey_col_to_pkey_table={},
    )
    recipe = NoriDFSRecipe(train_profile="v1", dense_history=True)

    median = model_module._median_rows_per_entity(task, train)

    assert median == rows_per_entity
    assert recipe.history_lags(median) == expected_lags
    assert NoriDFSRecipe(train_profile="v1").history_lags(median) == 5


def test_entity_density_ignores_unused_categories() -> None:
    task = SimpleNamespace(entity_col="entity")
    train = Table(
        pd.DataFrame(
            {
                "entity": pd.Categorical(
                    np.repeat([1, 2], DENSE_HISTORY_ROWS_PER_ENTITY),
                    categories=[1, 2, 3, 4],
                )
            }
        ),
        fkey_col_to_pkey_table={},
    )

    assert (
        model_module._median_rows_per_entity(task, train)
        == DENSE_HISTORY_ROWS_PER_ENTITY
    )


@pytest.mark.parametrize(
    ("train_rows", "median_rows_per_entity", "expected"),
    [
        (60_000, 2.0, False),
        (60_001, 1.0, False),
        (60_001, 2.0, True),
        (60_001, 31.0, True),
        (60_001, 32.0, False),
        (60_001, None, False),
    ],
)
def test_context_profile_uses_only_large_repeated_training_shape(
    train_rows: int, median_rows_per_entity: float | None, expected: bool
) -> None:
    recipe = NoriDFSRecipe(context_profile="v1")

    assert (
        model_module._context_profile_active(recipe, train_rows, median_rows_per_entity)
        is expected
    )
    assert not model_module._context_profile_active(
        NoriDFSRecipe(), train_rows, median_rows_per_entity
    )


@pytest.mark.parametrize(
    ("configured", "n_lags", "text_width", "expected"),
    [(87, 5, 16, 24), (87, 5, 0, 40), (20, 5, 0, 20), (87, 1, 16, 32), (87, 25, 16, 1)],
)
def test_context_budget_reserves_lag_and_text_width(
    configured: int, n_lags: int, text_width: int, expected: int
) -> None:
    assert (
        model_module._context_feature_budget(
            configured, n_lags=n_lags, text_width=text_width
        )
        == expected
    )


def test_context_profile_collapses_the_missing_arm_to_the_lean_budget() -> None:
    recipe = NoriDFSRecipe.from_config(model_module.NORIDFS_MISSING_CONFIG)

    wide = model_module._noridfs_featurizer(
        recipe, 2, context_profile=False, n_lags=5, text_width=16
    )
    narrow = model_module._noridfs_featurizer(
        recipe, 2, context_profile=True, n_lags=5, text_width=16
    )

    assert (wide._config.max_features, wide._config.max_direct_features) == (87, 1)
    assert wide._config.include_missing_in_modes
    assert (narrow._config.max_features, narrow._config.max_direct_features) == (24, 1)
    assert not narrow._config.include_missing_in_modes
    assert not narrow._config.include_missing_fractions
