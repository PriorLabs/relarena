"""TabPFN backend recipes and named fitting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from relbench.base import TaskType

from relarena.core.tfm import (
    FittedTFM,
    TFMSpec,
    default_device,
)
from relarena.core.tfm import fit_tfm as _fit_spec


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

    The local extra supplies TabPFN; importing model definitions does not load it.
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

    The api extra supplies tabpfn_client; it is imported when constructed.
    Fit and predict
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


TFM_REGISTRY: dict[str, TFMSpec] = {
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
    if device is None:
        device = default_device()
    return _fit_spec(
        df,
        y,
        task_type,
        spec=TFM_REGISTRY[tfm],
        seed=seed,
        device=device,
        max_train_samples=max_train_samples,
        max_predict_samples=max_predict_samples,
        overrides=overrides,
    )


__all__ = ["TFM_REGISTRY", "fit_tfm"]
