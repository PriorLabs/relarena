"""LightGBM baseline.

Combines two ideas:
  * **featurization** — the entity-only recipe from RelBench's LightGBM
    baseline (the entity's own columns + anchor timestamp; see
    `relarena.featurization.build_entity_features`);
  * **tuning** — TabArena's LightGBM hyperparameter search space, sampled by the
    RelArena harness under the caller's requested budget rather than LightGBM's
    own Optuna tuner.

Uses LightGBM's **native** `lgb.train` API (like AutoGluon / TabArena), so the
sampled config — which uses native LightGBM parameter names — is passed verbatim.
No PyTorch Frame / torch dependency.

No early stopping: `n_estimators` (num_boost_round) is part of the search space,
which fits the tune -> select -> refit protocol cleanly. `val_table` is unused by
the model — the harness uses it for selection.

Sources:
  * Hyperparameter search space adapted from TabArena's LightGBM config space:
    https://github.com/autogluon/tabarena — tabarena/models/lightgbm/generate.py
    (their ConfigSpace ranges, re-expressed as a seeded numpy sampler; n_estimators
    added since we tune rounds instead of early-stopping).
  * Featurization recipe from RelBench's LightGBM entity baseline (see
    `featurization.py`).
"""

from __future__ import annotations

import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Constant, Float, Integer
from relbench.base import Database, EntityTask, Table

from relarena.featurization import build_entity_features
from relarena.model import RelArenaModel
from relarena.models._shared.gbdt.lgb import fit_lgb, predict_lgb
from relarena.registry import register_model
from relarena.search_space import SearchSpace


def _lightgbm_config_space() -> ConfigurationSpace:
    """TabArena's LightGBM search space (native LightGBM param names) + n_estimators.

    Source: adapted from TabArena's LightGBM config space
    (https://github.com/autogluon/tabarena, `models/lightgbm/hpo.py`); we add
    `n_estimators` because we tune the number of boosting rounds rather than
    relying on early stopping.
    """
    return ConfigurationSpace(
        space=[
            Integer("n_estimators", (50, 1000), log=True),
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Float("feature_fraction", (0.4, 1.0)),
            Float("bagging_fraction", (0.7, 1.0)),
            Constant("bagging_freq", 1),
            Integer("num_leaves", (2, 200), log=True),
            Integer("min_data_in_leaf", (1, 64), log=True),
            Categorical("extra_trees", [False, True]),
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 2.0), log=True),
            Float("cat_smooth", (1e-3, 100.0), log=True),
            Integer("max_cat_to_onehot", (8, 100), log=True),
            Float("lambda_l1", (1e-4, 1.0)),
            Float("lambda_l2", (1e-4, 2.0)),
        ],
    )


#: The LightGBM model's search space: the config space above, with empty
#: `default_overrides` (-> LightGBM's own defaults, num_boost_round=100, etc.).
LIGHTGBM_SPACE = SearchSpace(space=_lightgbm_config_space(), default_overrides={})


@register_model(search_space=LIGHTGBM_SPACE)
class LightGBMModel(RelArenaModel):
    """Gradient-boosted trees on entity-only features.

    Tuned over `LIGHTGBM_SPACE` (TabArena's LightGBM space + `n_estimators`).
    """

    name = "lightgbm"

    # -- fit / predict --------------------------------------------------------

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Featurize the entity table and train LightGBM on `train_table`."""
        df, cat_cols = build_entity_features(task, db, train_table)

        # The config uses native LightGBM names, so it goes straight into params;
        # n_estimators is the boosting rounds (passed separately to lgb.train).
        params = dict(self.config)
        num_boost_round = int(params.pop("n_estimators", 100))
        self._fitted = fit_lgb(
            df,
            cat_cols,
            train_table.df[task.target_col],
            task.task_type,
            seed=seed,
            params=params,
            num_boost_round=num_boost_round,
        )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Featurize `table` and return the LightGBM predictions."""
        df, _ = build_entity_features(task, db, table)
        return predict_lgb(self._fitted, df)
