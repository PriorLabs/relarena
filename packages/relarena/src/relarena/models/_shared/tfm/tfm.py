"""Shared tabular-foundation-model (TFM) estimator core for the `rdblearn` baseline.

The `rdblearn` model feeds Deep Feature Synthesis features (`featurization/dfs.py`)
to a tabular foundation model and tunes over both DFS depth and *which* TFM is used.
This module owns the parts shared across TFMs and feature sources:

  * a small TFM **registry** mapping a name -> how to build its classifier/regressor,
    which task types it supports, and whether its backing package is importable. This
    is the single seam for adding TFMs — currently TabPFN v2 / v2.5 / v3 (local and
    hosted-API); a new TFM drops in here as one more entry;
  * seeded downsampling of the training set to each TFM's context-size cap
    (`TFMSpec.max_train_samples`);
  * fit, and predict that returns the shape `EntityTask.evaluate` expects (the
    sklearn-output reshaping is shared via `predict_contract`).

Categorical columns are handled by the TFM natively: we pass the feature frame as a
DataFrame and let TabPFN auto-detect categoricals (its preprocessing treats a column as
categorical below a cardinality threshold), rather than pre-encoding them (unlike
RDBLearn's `SafeLabelEncoder`) — hand-coding categoricals to a float matrix would only
hide them from TabPFN's categorical handling. We deliberately do *not* pass the
DFS-flagged columns as `categorical_features_indices`; tuning relational-data
preprocessing (incl. forcing categorical handling for moderate-cardinality columns) is
deliberately not done yet.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import torch
from relbench.base import TaskType

from relarena.models._shared.predict_contract import predict_to_contract


class SklearnClassifier(Protocol):
    """Minimal sklearn-classifier surface relarena uses (TabPFNClassifier-like)."""

    classes_: np.ndarray

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class SklearnRegressor(Protocol):
    """Minimal sklearn-regressor surface relarena uses (TabPFNRegressor-like)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# -- TFM registry ------------------------------------------------------------


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


def _make_tabpfn(
    version: str,
    *,
    regression: bool,
    device: Any,
    seed: int,
    **overrides: Any,
) -> Any:
    """Build a TabPFN estimator pinned to a version via `create_default_for_version`.

    The bare TabPFN constructor now defaults to v3; `create_default_for_version`
    selects the right checkpoint + version-appropriate defaults for v2 / v2.5, and
    `**overrides` (device, random_state, ignore_pretraining_limits, ...) pass through
    to the constructor.

    Lazy import — tabpfn lives in the rdblearn extra, and it is the one dependency
    under the Prior Labs License rather than a plain permissive one, so a core
    install stays clear of its attribution obligation (see `docs/licensing.md`).
    """
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model_version = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
        "v3": ModelVersion.V3,
    }[version]
    estimator_cls = TabPFNRegressor if regression else TabPFNClassifier
    return estimator_cls.create_default_for_version(
        model_version,
        device=device,
        random_state=seed,
        ignore_pretraining_limits=True,
        **overrides,
    )


def _sanitize_api_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy index arrays in `SUBSAMPLE_SAMPLES` to plain int lists.

    tabpfn_client pydantic-serializes the estimator config into the request body,
    which rejects numpy arrays — the pool contexts pass per-estimator context
    indices as arrays. The local TabPFN consumes arrays natively, so the
    conversion is scoped to the API path.
    """
    inference_config = overrides.get("inference_config")
    if inference_config is None:
        return overrides
    subsample = inference_config.get("SUBSAMPLE_SAMPLES")
    if not isinstance(subsample, list):
        return overrides
    return {
        **overrides,
        "inference_config": {
            **inference_config,
            "SUBSAMPLE_SAMPLES": [
                e.tolist() if isinstance(e, np.ndarray) else e for e in subsample
            ],
        },
    }


def _make_tabpfn_api(
    *,
    regression: bool,
    device: Any,
    seed: int,
    **overrides: Any,
) -> Any:
    """Build a TabPFN API-client estimator pinned to the v3 model.

    Lazy import — tabpfn_client lives in the tabpfn-rel-api extra. Fit and predict
    run server-side, so device is ignored and raw text columns are handled by the
    API.
    """
    from tabpfn_client import TabPFNClassifier as ApiClassifier
    from tabpfn_client import TabPFNRegressor as ApiRegressor

    del device
    overrides = _sanitize_api_overrides(overrides)
    estimator_cls = ApiRegressor if regression else ApiClassifier
    return estimator_cls(
        model_path="v3_default",
        random_state=seed,
        ignore_pretraining_limits=True,
        **overrides,
    )


def _tabpfn_spec(version: str, max_train_samples: int) -> TFMSpec:
    return TFMSpec(
        make_classifier=lambda **kw: _make_tabpfn(version, regression=False, **kw),
        make_regressor=lambda **kw: _make_tabpfn(version, regression=True, **kw),
        max_train_samples=max_train_samples,
    )


#: Name -> spec. The single seam for adding TFMs.
#: `max_train_samples` is each backend's fit limit. The RDBLearn paper
#: (arXiv:2602.18495) runs TabPFN v2, v2.5 and LimiX all under a **10k** limit,
#: downsampling above it — so those entries use 10k. `tabpfn-v3` is the
#: `tabpfn-rel-local`
#: backend (selected by its config, not swept by `rdblearn`); 100k = the context size
#: the reference sweeps ran it at, and `n_preprocessing_jobs=-1` parallelizes its
#: heavier preprocessing. `tabpfn-v3-api` is the same v3 model served by the TabPFN
#: API (no GPU needed); 100k matches the local entry and sits well inside the API's v3
#: train limits, and the API handles raw text columns server-side.
TFM_REGISTRY: dict[str, TFMSpec] = {
    "tabpfn-v2": _tabpfn_spec("v2", max_train_samples=10_000),
    "tabpfn-v2.5": _tabpfn_spec("v2.5", max_train_samples=10_000),
    "tabpfn-v3": TFMSpec(
        make_classifier=lambda **kw: _make_tabpfn(
            "v3", regression=False, n_preprocessing_jobs=-1, **kw
        ),
        make_regressor=lambda **kw: _make_tabpfn(
            "v3", regression=True, n_preprocessing_jobs=-1, **kw
        ),
        max_train_samples=100_000,
    ),
    "tabpfn-v3-api": TFMSpec(
        make_classifier=lambda **kw: _make_tabpfn_api(regression=False, **kw),
        make_regressor=lambda **kw: _make_tabpfn_api(regression=True, **kw),
        max_train_samples=100_000,
        supports_text=True,
    ),
}


def default_device() -> str:
    """Return `"cuda"` if a GPU is visible to torch, else `"cpu"`."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# -- downsampling ------------------------------------------------------------


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


# -- fit / predict -----------------------------------------------------------


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
    tfm: str,
    seed: int,
    device: Any = None,
    max_train_samples: int | None = None,
    max_predict_samples: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> FittedTFM:
    """Downsample `df` and fit the named TFM on it.

    `df` is the already-typed feature frame from `build_dfs_features` (numeric
    floats + object categoricals); TabPFN auto-detects categoricals from it (see the
    module docstring) — we do not pass `categorical_features_indices`. The training
    rows are capped (seeded) at `max_train_samples` if given, else the TFM's own
    context cap (`spec.max_train_samples`); `overrides` are additional
    estimator-constructor arguments. `max_predict_samples` is an explicit
    caller-owned cap on rows per estimator prediction call; ordinary TFM callers
    leave it unset.
    """
    spec = TFM_REGISTRY[tfm]
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
    `relarena.models._shared.predict_contract.predict_to_contract`.

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
