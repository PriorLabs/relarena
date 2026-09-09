"""Unit tests for the recency-pool index math + the context strategies (stub TFM)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models._shared.tfm import tfm
from relarena.models.tabpfn_rel.context import (
    ContextStrategy,
    HardPoolContext,
    RandomContext,
    SoftPoolContext,
    hard_pool_subsample_indices,
    soft_pool_subsample_indices,
)

# -- pool index math ---------------------------------------------------------


def test_pool_indices_shape_and_within_pool() -> None:
    t = np.arange(100)  # ascending time -> higher index = more recent
    for fn in (
        lambda: soft_pool_subsample_indices(
            t, K=8, M=40, n_estimators=4, tau_half_frac=0.1, seed=0
        ),
        lambda: hard_pool_subsample_indices(t, K=8, M=40, n_estimators=4, seed=0),
    ):
        idx_list = fn()
        assert idx_list is not None and len(idx_list) == 4  # one array per estimator
        for arr in idx_list:
            assert len(arr) == 8  # K rows each
            assert len(set(arr.tolist())) == 8  # without replacement
            assert arr.min() >= 0 and arr.max() < 100


def test_hard_pool_draws_only_from_most_recent_M() -> None:
    t = np.arange(100)
    idx_list = hard_pool_subsample_indices(t, K=5, M=20, n_estimators=3, seed=1)
    assert idx_list is not None
    for arr in idx_list:  # pool = the 20 most-recent rows = indices 80..99
        assert arr.min() >= 80


def test_soft_pool_concentrates_on_recent_rows_with_small_tau() -> None:
    t = np.arange(1000)
    idx_list = soft_pool_subsample_indices(
        t, K=20, M=100, n_estimators=5, tau_half_frac=0.02, seed=0
    )
    assert idx_list is not None
    # With a very small tau the pool is dominated by recent rows.
    assert np.concatenate(idx_list).mean() > 700


def test_pool_indices_none_when_n_le_m() -> None:
    t = np.arange(30)
    assert (
        soft_pool_subsample_indices(
            t, K=8, M=40, n_estimators=4, tau_half_frac=0.1, seed=0
        )
        is None
    )
    assert hard_pool_subsample_indices(t, K=8, M=40, n_estimators=4, seed=0) is None


def test_pool_indices_are_seeded_deterministic() -> None:
    t = np.arange(100)
    a = soft_pool_subsample_indices(
        t, K=8, M=40, n_estimators=4, tau_half_frac=0.1, seed=7
    )
    b = soft_pool_subsample_indices(
        t, K=8, M=40, n_estimators=4, tau_half_frac=0.1, seed=7
    )
    assert a is not None and b is not None
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_hard_and_soft_pool_handle_datetime_input() -> None:
    # Real cutoff timestamps arrive as datetime64; _to_int64_ns must sort them.
    t = np.array(
        [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(60)]
    )
    idx = hard_pool_subsample_indices(t, K=5, M=20, n_estimators=2, seed=0)
    assert idx is not None
    for arr in idx:
        assert arr.min() >= 40  # most-recent 20 of 60


# -- context strategies (stub TFM, no real TabPFN) ---------------------------


class _StubClassifier:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_StubClassifier":
        self.classes_ = np.unique(y)
        self.n_train_ = len(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = np.arange(1, len(self.classes_) + 1, dtype=float)
        return np.tile(cols / cols.sum(), (len(X), 1))


@pytest.fixture
def capture() -> Iterator[dict[str, object]]:
    """Register a kwargs-recording stub TFM (3-field TFMSpec); yield the ctor kwargs."""
    captured: dict[str, object] = {}

    def _make(**kw: object) -> _StubClassifier:
        captured.update(kw)
        return _StubClassifier()

    tfm.TFM_REGISTRY["capture"] = tfm.TFMSpec(
        make_classifier=_make, make_regressor=_make, max_train_samples=10
    )
    try:
        yield captured
    finally:
        del tfm.TFM_REGISTRY["capture"]


def _df_y(n: int) -> tuple[pd.DataFrame, pd.Series]:
    return pd.DataFrame({"f": np.arange(float(n))}), pd.Series([0, 1] * (n // 2))


def test__hard_pool_context__fit__fits_union_with_remapped_indices(
    capture: dict[str, object],
) -> None:
    df, y = _df_y(100)
    t = np.arange(100)
    strategy = HardPoolContext(
        subsample_samples=8, n_estimators=4, pool_inflation=4.0, subsample_rows=None
    )
    fitted = strategy.fit(
        df, y, TaskType.BINARY_CLASSIFICATION, tfm="capture", seed=0, context_time=t
    )
    ss = capture["inference_config"]["SUBSAMPLE_SAMPLES"]  # type: ignore[index]
    assert isinstance(ss, list) and len(ss) == 4 and all(len(a) == 8 for a in ss)
    assert capture["n_estimators"] == 4

    # The TFM fits only the union of the per-estimator contexts (bounds the frame),
    # and each estimator's SUBSAMPLE_SAMPLES index back into the original rows.
    idx = hard_pool_subsample_indices(t, K=8, M=32, n_estimators=4, seed=0)
    union = np.unique(np.concatenate(idx))
    assert fitted.estimator.n_train_ == len(union)
    for got, want in zip(ss, idx):
        assert np.array_equal(union[np.asarray(got)], want)


def test__pool_context__subsample_rows__caps_before_pooling(
    capture: dict[str, object],
) -> None:
    df, y = _df_y(100)
    strategy = HardPoolContext(
        subsample_samples=4, n_estimators=2, pool_inflation=4.0, subsample_rows=30
    )
    fitted = strategy.fit(
        df,
        y,
        TaskType.BINARY_CLASSIFICATION,
        tfm="capture",
        seed=0,
        context_time=np.arange(100),
    )
    # Cap to 30 rows, then pool: the fitted frame is the union of the 2 estimators'
    # K=4 draws from the M=16 pool — at most 8 unique rows, never more than the cap.
    assert fitted.estimator.n_train_ <= 8


def test__pool_context__degenerate_pool__falls_back_to_int(
    capture: dict[str, object],
) -> None:
    df, y = _df_y(20)  # N=20 <= M = 8*4 = 32 -> degenerate
    strategy = SoftPoolContext(
        subsample_samples=8, n_estimators=4, pool_inflation=4.0, subsample_rows=None
    )
    strategy.fit(
        df,
        y,
        TaskType.BINARY_CLASSIFICATION,
        tfm="capture",
        seed=0,
        context_time=np.arange(20),
    )
    assert capture["inference_config"] == {"SUBSAMPLE_SAMPLES": 8}  # int fallback


def test__random_context__fit__defers_to_base_fit_tfm(
    capture: dict[str, object],
) -> None:
    df, y = _df_y(100)
    RandomContext().fit(
        df, y, TaskType.BINARY_CLASSIFICATION, tfm="capture", seed=0, context_time=None
    )
    assert "inference_config" not in capture  # no pool overrides; plain downsample fit


# -- from_config: exactly one valid mode -------------------------------------


def test__from_config__per_name__selects_one_strategy() -> None:
    assert isinstance(ContextStrategy.from_config({}), RandomContext)  # default
    hard = ContextStrategy.from_config(
        {"context_strategy": "hard_pool", "subsample_samples": 8}
    )
    assert isinstance(hard, HardPoolContext) and hard.subsample_samples == 8
    soft = ContextStrategy.from_config(
        {"context_strategy": "soft_pool", "subsample_samples": 8, "tau_half_frac": 0.2}
    )
    assert isinstance(soft, SoftPoolContext) and soft.tau_half_frac == 0.2


def test__from_config__pool_without_k__raises() -> None:
    with pytest.raises(ValueError, match="subsample_samples"):
        ContextStrategy.from_config({"context_strategy": "hard_pool"})


def test__from_config__unknown_strategy__raises() -> None:
    with pytest.raises(ValueError, match="unknown context_strategy"):
        ContextStrategy.from_config({"context_strategy": "bogus"})
