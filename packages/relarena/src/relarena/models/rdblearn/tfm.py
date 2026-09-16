"""TabPFN backend recipes for RDBLearn."""

from __future__ import annotations

from typing import Any

from relarena.core.tfm import TFMSpec


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
    }[version]
    estimator_cls = TabPFNRegressor if regression else TabPFNClassifier
    return estimator_cls.create_default_for_version(
        model_version,
        device=device,
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


TFM_REGISTRY: dict[str, TFMSpec] = {
    "tabpfn-v2": _tabpfn_spec("v2", max_train_samples=10_000),
    "tabpfn-v2.5": _tabpfn_spec("v2.5", max_train_samples=10_000),
}


__all__ = ["TFM_REGISTRY"]
