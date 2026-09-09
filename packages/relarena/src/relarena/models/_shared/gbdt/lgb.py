"""LightGBM estimator core for the entity-only baseline.

The public `lightgbm` model trains LightGBM's native `lgb.train` API on a prepared
tabular feature matrix. The estimator mechanics live here so the model adapter can
stay focused on RelArena's fit/predict contract:

  * objective selection from the task type;
  * freezing categorical vocabularies on the training frame so val/test encode
    consistently (unseen categories -> NaN, treated as missing by LightGBM);
  * label construction and training;
  * prediction (-> `(N,)`).

The model supplies the feature matrix + its native-parameter config and gets back a
`FittedLGB` to keep for prediction. No PyTorch Frame / torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from relbench.base import TaskType

#: Native LightGBM objective per supported entity task type. Regression uses the L1
#: (`regression_l1`) objective so the model optimizes MAE, the regression primary
#: metric: L1 targets the conditional median, whereas the default `regression` (L2)
#: objective targets the mean and underperforms a constant-median predictor on MAE
#: for skewed targets.
_OBJECTIVE = {
    TaskType.REGRESSION: "regression_l1",
    TaskType.BINARY_CLASSIFICATION: "binary",
}


@dataclass
class FittedLGB:
    """A trained booster plus the state needed to encode val/test consistently.

    `cat_dtypes` pins each categorical column's vocabulary to what training saw, and
    `feature_cols` fixes column order, so `predict_lgb` reproduces the exact
    training-time encoding on a new split.
    """

    booster: Any
    cat_cols: list[str]
    cat_dtypes: dict[str, pd.CategoricalDtype]
    feature_cols: list[str]


def fit_lgb(
    df: pd.DataFrame,
    cat_cols: list[str],
    y: pd.Series,
    task_type: TaskType,
    *,
    seed: int,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 100,
) -> FittedLGB:
    """Train a native LightGBM booster on a prepared `(df, cat_cols)` matrix.

    `params` are native LightGBM parameter names, passed verbatim (the caller owns
    the search space). `df` is mutated in place to apply the frozen categorical
    dtypes.
    """
    import lightgbm as lgb

    # Freeze categorical vocabularies on the training data (see module docstring).
    cat_dtypes = {
        c: pd.CategoricalDtype(categories=df[c].astype("category").cat.categories)
        for c in cat_cols
    }
    for c in cat_cols:
        df[c] = df[c].astype(cat_dtypes[c])
    feature_cols = list(df.columns)

    if task_type == TaskType.BINARY_CLASSIFICATION:
        label = y.to_numpy()
    elif task_type == TaskType.REGRESSION:
        label = y.to_numpy(dtype=float)
    else:
        raise ValueError(f"LightGBM does not support task type {task_type}.")

    full_params = dict(params or {})
    full_params.update(objective=_OBJECTIVE[task_type], verbosity=-1, seed=seed)

    # DFS names describe the full aggregation expression and can contain JSON
    # punctuation (for example quotes and brackets), which LightGBM rejects in
    # feature names.  LightGBM does not use names to define the fitted function,
    # so train with stable positional names and retain the caller-facing names in
    # `FittedLGB` for prediction-time alignment.
    lgb_cols = [f"f{i}" for i in range(len(feature_cols))]
    lgb_cat_cols = [lgb_cols[feature_cols.index(c)] for c in cat_cols]
    lgb_df = df.copy(deep=False)
    lgb_df.columns = lgb_cols

    dtrain = lgb.Dataset(
        lgb_df, label=label, categorical_feature=lgb_cat_cols or "auto"
    )
    booster = lgb.train(full_params, dtrain, num_boost_round=num_boost_round)
    return FittedLGB(booster, cat_cols, cat_dtypes, feature_cols)


def predict_lgb(fitted: FittedLGB, df: pd.DataFrame) -> np.ndarray:
    """Predict with a `FittedLGB` on a new feature frame.

    Applies the frozen categorical dtypes and the training column order, then calls
    the native booster (which returns the shape `EntityTask.evaluate` expects).
    """
    for c in fitted.cat_cols:
        df[c] = df[c].astype(fitted.cat_dtypes[c])
    df = df.reindex(columns=fitted.feature_cols)
    df.columns = [f"f{i}" for i in range(len(fitted.feature_cols))]
    return fitted.booster.predict(df)
