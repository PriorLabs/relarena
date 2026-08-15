"""DFS + tabular foundation model baseline — RDBLearn proper.

Combines:
  * **featurization** — multi-hop Deep Feature Synthesis over the foreign-key graph
    (`relarena.featurization.build_dfs_features`, with the depth cache),
    plus target-history augmentation (past-label aggregates), temporal-diff
    features, and the anchor columns (entity key + cutoff-time calendar features);
  * **search space** — an explicit grid over **(which tabular foundation model) ×
    (DFS depth)** (`RDBLEARN_SPACE`, bound to the model in the registry). Each config
    is `{"tfm", "max_depth"}`, enumerated by the harness; shallower depths reuse the
    cached deepest matrix, so the DFS cost is paid once per split regardless of the TFM
    dimension.

The estimator is a tabular foundation model (TabPFN v2 / v2.5); see
`_shared/tfm/tfm.py` for the TFM registry and the downsample -> fit/predict core
(the TFM handles categoricals natively). This is RDBLearn proper
(https://github.com/HKUSHXLab/rdblearn) — DFS features + a foundation model.

*Note:* We currently intentionally exclude LimiX, which is the third TFM RDBLearn
tunes over. This is because LimiX does not provide a PyPI package for installation,
requiring us to vendor their code in addition to vendoring RDBLearn itself. This
likely leads to the RDBLearn baseline performing worse than it could if LimiX were
included (at the cost of additional runtime).

One deliberate deviation from upstream RDBLearn's preprocessing: it label-encodes
categoricals and then runs AutoGluon's `AutoMLPipelineFeatureGenerator` over the
result, whereas here the feature frame reaches the TFM as-is, so TabPFN does its own
categorical detection and NaN handling (`_shared/tfm/tfm.py` has the why — upstream
encodes because its backends take numpy arrays, a constraint a TabPFN-only grid does
not have). The generator's datetime expansion is covered natively: the anchor cutoff
gets the same year / month / day / dayofweek decomposition plus an epoch value, and
DFS-aggregated timestamps arrive as time-until-cutoff diffs. Not replicated: its
rare-category collapsing and useless/duplicate-column pruning.
"""

from __future__ import annotations

import os

import numpy as np
from relbench.base import Database, EntityTask, Table

from relarena.featurization import DFS_MAX_DEPTH, build_dfs_features
from relarena.model import RelArenaModel
from relarena.models._shared.tfm.tfm import (
    fit_tfm,
    predict_tfm,
)
from relarena.registry import register_model
from relarena.search_space import SearchSpace

_MIN_DEPTH = 2

#: Tabular foundation models to sweep (names in the shared `TFM_REGISTRY`); both
#: ship in the `tabpfn` package (a core dependency) and run under the RDBLearn
#: paper's 10k fit limit. The paper (arXiv:2602.18495) sweeps exactly TabPFN
#: v2 / v2.5 (and LimiX, which
#: has no pip package — it / TabPFN-v3 plug in here once registered).
_TFMS = ("tabpfn-v2", "tabpfn-v2.5")

#: RDBLearn's TabPFN v2/v2.5 inference mode does not safely chunk large test
#: matrices itself: its 32,768-row default exceeded a 96 GiB RTX Pro 6000 in the
#: full sweep. This workaround belongs to RDBLearn rather than the shared TFM
#: registry, so TabPFN-Rel (including local v3) keeps its native inference behavior.
_MAX_PREDICT_SAMPLES = 8192


def _configure_prediction_batching() -> int | None:
    """Apply RDBLearn's v2/v2.5 prediction cap and return the external batch size."""
    value = int(
        os.environ.setdefault("TABPFN_MAX_BATCHED_TEST_ROWS", str(_MAX_PREDICT_SAMPLES))
    )
    if value <= 0:
        return None

    # TabPFN settings may already have been instantiated by another import. Update
    # the live object as well as the environment used during a fresh import.
    from tabpfn.settings import settings

    settings.tabpfn.max_batched_test_rows = value
    return value


#: Explicit grid over TFM × DFS depth (TFMs in `_TFMS` order, depth ascending).
#: Ascending so the default (depth 2) comes first and survives budget truncation —
#: `SearchSpace.configs` keeps only the first `n_trials`. Grid *order* does not
#: affect the DFS depth cache; see `featurization/dfs.py`
#: (`_DepthCache.full_matrix`) for how the search interacts with the cache.
#: Enumerated (not sampled): a small grid.
RDBLEARN_SPACE = SearchSpace(
    default_overrides={"tfm": _TFMS[0], "max_depth": 2},
    fixed_grid=[
        {"tfm": tfm, "max_depth": d}
        for d in range(_MIN_DEPTH, DFS_MAX_DEPTH + 1)
        for tfm in _TFMS
    ],
)


@register_model(search_space=RDBLEARN_SPACE)
class RDBLearnModel(RelArenaModel):
    """DFS features + a tabular foundation model, tuned over TFM × DFS depth."""

    name = "rdblearn"

    #: Upstream RDBLearn does not refit on train+val; the final fit stays train-only.
    refit_on_full_data = False

    #: Depth at which the cached DFS matrix is computed (shallower configs slice it).
    MAX_DEPTH = DFS_MAX_DEPTH

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
        """Build DFS features at the configured depth and fit the chosen TFM."""
        self._tfm = self.config.get("tfm", _TFMS[0])
        self._depth = int(self.config.get("max_depth", 2))

        self._history_table = train_table if task.time_col else None

        df, _ = build_dfs_features(
            task,
            db,
            train_table,
            depth=self._depth,
            max_depth=self.MAX_DEPTH,
            history_table=self._history_table,
            keep_anchor_columns=True,
            cache=self.cache,
            run_identity=self.run_identity,
        )
        if df.shape[1] == 0:
            # No DFS features at this depth (e.g. depth 0/1 produce none); nothing to
            # train on. Surface a clean error — the harness skips this trial.
            raise ValueError(f"DFS produced no features at depth {self._depth}.")

        self._fitted = fit_tfm(
            df,
            train_table.df[task.target_col],
            task.task_type,
            tfm=self._tfm,
            seed=seed,
            max_predict_samples=(
                _configure_prediction_batching() if self._tfm in _TFMS else None
            ),
        )

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        """Build DFS features for `table` and return the TFM predictions."""
        df, _ = build_dfs_features(
            task,
            db,
            table,
            depth=self._depth,
            max_depth=self.MAX_DEPTH,
            history_table=self._history_table,
            keep_anchor_columns=True,
            cache=self.cache,
            run_identity=self.run_identity,
        )
        return predict_tfm(self._fitted, df)
