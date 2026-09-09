"""Shared estimator fitting, sampling and prediction mechanics."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import torch
from relbench.base import TaskType

from relarena_core.predict_contract import predict_to_contract


class SklearnClassifier(Protocol):
    """Minimal sklearn-classifier surface relarena uses (TabPFNClassifier-like)."""

    classes_: np.ndarray

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Fit the classifier on the feature frame and labels."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return one probability column per fitted class."""
        ...


class SklearnRegressor(Protocol):
    """Minimal sklearn-regressor surface relarena uses (TabPFNRegressor-like)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Fit the regressor on the feature frame and targets."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one numeric prediction per input row."""
        ...


@dataclass(frozen=True)
class TFMSpec:
    """How to build one tabular foundation model.

    Every TFM is assumed to support all entity task types, so there is no per-TFM
    task-type gating. `make_classifier` / `make_regressor` take keyword overrides
    (`device`, `seed`, ...) and return an estimator satisfying
    `SklearnClassifier` / `SklearnRegressor` respectively.
    `max_train_samples` is this TFM's training-row cap before fitting — its
    supported context size (TabPFN v2 ~10k, v2.5 ~50k) — applied by `fit_tfm`.
    `supports_text` marks estimators that handle raw text columns themselves.
    """

    make_classifier: Callable[..., SklearnClassifier]
    make_regressor: Callable[..., SklearnRegressor]
    max_train_samples: int
    supports_text: bool = False


def default_device() -> str:
    """Return `"cuda"` if a GPU is visible to torch, else `"cpu"`."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _downsample_indices(
    y: np.ndarray, task_type: TaskType, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of a <= `max_samples` subset of rows (no-op when already small).

    Regression: a uniform random subset. Classification: keep at least one row per
    class, then fill the remaining budget uniformly at random. Seeded via `rng` for
    reproducibility. Adapted from RDBLearn's `_downsample` (non-stratified path).
    """
    n = len(y)
    if n <= max_samples:
        return np.arange(n)

    if task_type == TaskType.REGRESSION:
        return rng.choice(n, max_samples, replace=False)

    selected: list[int] = []
    for label in np.unique(y):
        class_idx = np.where(y == label)[0]
        selected.append(int(rng.choice(class_idx, 1)[0]))
    selected = list(dict.fromkeys(selected))  # de-dup (one per class)

    remaining = max_samples - len(selected)
    if remaining > 0:
        mask = np.ones(n, dtype=bool)
        mask[selected] = False
        eligible = np.where(mask)[0]
        extra = rng.choice(eligible, min(remaining, len(eligible)), replace=False)
        out = np.concatenate([np.array(selected, dtype=int), extra])
    else:
        out = np.array(selected[:max_samples], dtype=int)
    rng.shuffle(out)
    return out


@dataclass
class FittedTFM:
    """A fitted TFM plus the state needed to score val/test consistently."""

    estimator: Any
    feature_cols: list[str]
    task_type: TaskType
    max_predict_samples: int | None = None


def fit_tfm(
    df: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    *,
    spec: TFMSpec,
    seed: int,
    device: Any = None,
    max_train_samples: int | None = None,
    max_predict_samples: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> FittedTFM:
    """Downsample `df` and fit the supplied estimator specification.

    `df` is the already-typed feature frame from `build_dfs_features` (numeric
    floats + object categoricals); TabPFN auto-detects categoricals from it (see the
    module docstring) — we do not pass `categorical_features_indices`. The training
    rows are capped (seeded) at `max_train_samples` if given, else the TFM's own
    context cap (`spec.max_train_samples`); `overrides` are additional
    estimator-constructor arguments. `max_predict_samples` is an explicit
    caller-owned cap on rows per estimator prediction call; ordinary TFM callers
    leave it unset.
    """
    if device is None:
        device = default_device()
    cap = max_train_samples if max_train_samples is not None else spec.max_train_samples
    rng = np.random.default_rng(seed)

    feature_cols = list(df.columns)

    y_arr = y.to_numpy()
    idx = _downsample_indices(y_arr, task_type, cap, rng)
    X = df.iloc[idx]
    y_arr = y_arr[idx]

    kwargs = dict(device=device, seed=seed, **(overrides or {}))
    if task_type == TaskType.REGRESSION:
        estimator = spec.make_regressor(**kwargs)
        y_arr = y_arr.astype(float)
    else:
        estimator = spec.make_classifier(**kwargs)
    estimator.fit(X, y_arr)

    return FittedTFM(estimator, feature_cols, task_type, max_predict_samples)


def _predict_tfm_frame(fitted: FittedTFM, frame: pd.DataFrame) -> np.ndarray:
    if fitted.task_type == TaskType.REGRESSION:
        predict = fitted.estimator.predict
        params = inspect.signature(predict).parameters.values()
        if any(
            p.name == "output_type" or p.kind is inspect.Parameter.VAR_KEYWORD
            for p in params
        ):
            return np.asarray(predict(frame, output_type="median"), dtype=float)
    return predict_to_contract(fitted.estimator, frame, fitted.task_type)


def predict_tfm(fitted: FittedTFM, df: pd.DataFrame) -> np.ndarray:
    """Predict with a `FittedTFM` on a new feature frame.

    Reindexes to the training column order (so the TFM sees the same schema), then
    delegates the sklearn-output -> evaluate-contract reshaping to
    `relarena_core.predict_contract.predict_to_contract`.

    Regression requests `output_type="median"` when the estimator supports it —
    an explicit output_type parameter, or a **kwargs passthrough: the primary
    regression metric is MAE, and the median is its optimal point prediction.
    """
    X = df.reindex(columns=fitted.feature_cols)
    batch_size = fitted.max_predict_samples
    if batch_size is None or len(X) <= batch_size:
        return _predict_tfm_frame(fitted, X)
    return np.concatenate(
        [
            _predict_tfm_frame(fitted, X.iloc[i : i + batch_size])
            for i in range(0, len(X), batch_size)
        ]
    )
