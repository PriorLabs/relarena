"""Context-selection strategies for the `tabpfn-rel` TFM fit.

The TFM learns in-context from the train rows; *which* rows it sees per ensemble
estimator is the **context strategy**. `tabpfn-rel` supports exactly one of:

  * `random`    — a uniform seeded downsample to the TFM's context cap (the base
    `rdblearn` behavior; no extra parameters);
  * `hard_pool` — each estimator draws `K` rows from the `M` most-recent rows;
  * `soft_pool` — each estimator draws `K` rows from an `M`-row pool sampled
    with recency-decaying weights (half-weight at the `tau_half_frac` percentile).

`ContextStrategy.from_config` turns the single `context_strategy` config
value (plus that mode's own knobs) into one strategy object — there is no way to
combine modes, and a missing required knob (e.g. the pool `K`) raises here.
`ContextStrategy.fit` then builds and fits the TFM; the pool strategies fit on
the **union** of the per-estimator contexts (bounding the frame the TFM stores) and
pass each estimator's rows as `inference_config={"SUBSAMPLE_SAMPLES": [...]}`.
Categoricals are auto-inferred by the TFM throughout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from relbench.base import TaskType

from relarena.models._shared.tfm.tfm import FittedTFM, fit_tfm

#: Default ensemble size for the pool strategies when the config pins none.
DEFAULT_POOL_N_ESTIMATORS = 8


def _to_int64_ns(t: np.ndarray) -> np.ndarray:
    """Coerce a per-row cutoff-timestamp array to int64 nanoseconds (sortable)."""
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").view("int64")
    if t.dtype == object:
        return pd.to_datetime(t).astype("int64").to_numpy()
    return t.astype("int64")


def soft_pool_subsample_indices(
    t: np.ndarray,
    *,
    K: int,
    M: int,
    n_estimators: int,
    tau_half_frac: float,
    seed: int,
) -> list[np.ndarray] | None:
    """Recency-weighted per-estimator context subsample (`soft_pool`).

    Stage 1 (shared across estimators): draw an `M`-row pool from the `N` context
    rows via Efraimidis-Spirakis weighted sampling, with weight
    `exp(-ln(2)/tau_half_frac · rank_frac)` where `rank_frac` is in `[0, 1]`
    (0 = most recent by `t`). `tau_half_frac=0.1` halves the weight at the 10th
    recency percentile. Stage 2 (per estimator): draw `K` rows uniformly without
    replacement from the pool (pairwise overlap ≈ `K²/M`).

    Returns `None` when `N <= M` (the pool would be everything, so recency
    weighting is inert) — the caller then falls back to TabPFN's uniform path.
    """
    n = len(t)
    if n <= M:
        return None
    rng = np.random.default_rng(seed)
    ti = _to_int64_ns(t)
    order = np.argsort(-ti, kind="stable")  # order[0] = most-recent row index
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)  # ranks[row] = recency rank (0 = most recent)
    r_frac = ranks / max(n - 1, 1)
    lam = np.log(2.0) / max(tau_half_frac, 1e-12)
    weights = np.exp(-lam * r_frac)  # strictly positive for every row
    log_u = np.log(rng.random(n))  # strictly negative
    keys = log_u / weights  # Efraimidis-Spirakis key; larger = more likely
    pool = np.argpartition(keys, -M)[-M:]
    return [rng.choice(pool, size=K, replace=False) for _ in range(n_estimators)]


def hard_pool_subsample_indices(
    t: np.ndarray, *, K: int, M: int, n_estimators: int, seed: int
) -> list[np.ndarray] | None:
    """Most-recent-`M` per-estimator context subsample (`hard_pool`).

    Stage 1: the pool is the `M` most-recent rows by `t` (deterministic; ties
    broken by a stable sort). Stage 2: each estimator draws `K` rows uniformly
    without replacement from the pool. Returns `None` when `N <= M` (caller falls
    back to TabPFN's uniform path).
    """
    n = len(t)
    if n <= M:
        return None
    ti = _to_int64_ns(t)
    order = np.argsort(-ti, kind="stable")  # order[0] = most recent
    pool = order[:M]
    rng = np.random.default_rng(seed)
    return [rng.choice(pool, size=K, replace=False) for _ in range(n_estimators)]


class ContextStrategy(ABC):
    """How `tabpfn-rel` picks the TFM's in-context training rows (see module doc)."""

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        task_type: TaskType,
        *,
        tfm: str,
        seed: int,
        context_time: np.ndarray | None,
    ) -> FittedTFM:
        """Fit `tfm` on the featurized frame under this strategy.

        `context_time` is the per-row cutoff timestamp (row-aligned with `df` /
        `y`), i.e. each candidate context row's prediction time — what the pool
        strategies rank rows by to over-weight recent ones. `None` (or unused) for
        `random`, which is recency-agnostic; the pools require it.
        """

    @staticmethod
    def from_config(config: dict[str, Any]) -> "ContextStrategy":
        """Build the single strategy the config selects (default `random`).

        Raises on an unknown name, and (for the pools) on a missing
        `subsample_samples` — so a run commits to exactly one valid mode up front.
        """
        builders: dict[str, Callable[[dict[str, Any]], ContextStrategy]] = {
            "random": RandomContext.from_config,
            "hard_pool": HardPoolContext.from_config,
            "soft_pool": SoftPoolContext.from_config,
        }
        name = config.get("context_strategy", "random")
        if name not in builders:
            raise ValueError(
                f"unknown context_strategy {name!r}; choose one of {sorted(builders)}"
            )
        return builders[name](config)


@dataclass(frozen=True)
class RandomContext(ContextStrategy):
    """Uniform seeded downsample to the TFM's context cap (the `rdblearn` default)."""

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        task_type: TaskType,
        *,
        tfm: str,
        seed: int,
        context_time: np.ndarray | None = None,
    ) -> FittedTFM:
        """Defer to the base `fit_tfm` (it caps rows itself); recency unused."""
        return fit_tfm(df, y, task_type, tfm=tfm, seed=seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RandomContext":
        """Random context takes no parameters."""
        return cls()


@dataclass(frozen=True)
class _PoolContext(ContextStrategy):
    """Shared recency-pool fit; subclasses only differ in how the pool is drawn.

    With `K = subsample_samples` and `M = round(K·pool_inflation)`: optionally cap
    the frame to `subsample_rows` random rows, draw per-estimator contexts from the
    recency pool, then fit on their **union** (bounding the frame the TFM stores) with
    each estimator's rows remapped in and passed as `SUBSAMPLE_SAMPLES`. A degenerate
    pool (`N <= M`) has no union — the TFM fits the whole frame and subsamples `K`.
    """

    subsample_samples: int
    n_estimators: int
    pool_inflation: float
    subsample_rows: int | None

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        task_type: TaskType,
        *,
        tfm: str,
        seed: int,
        context_time: np.ndarray | None,
    ) -> FittedTFM:
        """Cap, draw the pool, fit the union (see class doc)."""
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True)
        t = np.asarray(context_time)

        if self.subsample_rows is not None and len(df) > self.subsample_rows:
            keep = np.random.default_rng(seed).permutation(len(df))[
                : self.subsample_rows
            ]
            df = df.iloc[keep].reset_index(drop=True)
            y = y.iloc[keep].reset_index(drop=True)
            t = t[keep]

        m = int(round(self.subsample_samples * self.pool_inflation))
        idx = self._draw_pool(t, m=m, seed=seed)
        if idx is None:
            subsample_value: Any = self.subsample_samples
        else:
            union = np.unique(np.concatenate(idx))
            df = df.iloc[union].reset_index(drop=True)
            y = y.iloc[union].reset_index(drop=True)
            subsample_value = [np.searchsorted(union, e) for e in idx]

        overrides = {
            "n_estimators": self.n_estimators,
            "inference_config": {"SUBSAMPLE_SAMPLES": subsample_value},
        }
        return fit_tfm(
            df,
            y,
            task_type,
            tfm=tfm,
            seed=seed,
            max_train_samples=len(df),
            overrides=overrides,
        )

    @abstractmethod
    def _draw_pool(
        self, t: np.ndarray, *, m: int, seed: int
    ) -> list[np.ndarray] | None:
        """Per-estimator context indices into `t` (`None` if pool = all rows)."""

    @staticmethod
    def _shared(config: dict[str, Any]) -> dict[str, Any]:
        """Parse the knobs every pool shares; require the pool size `K`."""
        if config.get("subsample_samples") is None:
            raise ValueError(
                f"context_strategy={config.get('context_strategy')!r} requires "
                f"subsample_samples (K)"
            )
        rows = config.get("subsample_rows")
        return dict(
            subsample_samples=int(config["subsample_samples"]),
            n_estimators=int(config.get("n_estimators") or DEFAULT_POOL_N_ESTIMATORS),
            pool_inflation=float(config.get("pool_inflation", 4.0)),
            subsample_rows=None if rows is None else int(rows),
        )


@dataclass(frozen=True)
class HardPoolContext(_PoolContext):
    """Each estimator draws `K` rows from the `M` most-recent rows."""

    def _draw_pool(
        self, t: np.ndarray, *, m: int, seed: int
    ) -> list[np.ndarray] | None:
        return hard_pool_subsample_indices(
            t, K=self.subsample_samples, M=m, n_estimators=self.n_estimators, seed=seed
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HardPoolContext":
        """Build from the shared pool knobs."""
        return cls(**cls._shared(config))


@dataclass(frozen=True)
class SoftPoolContext(_PoolContext):
    """Each estimator draws `K` rows from a recency-weighted `M`-row pool."""

    tau_half_frac: float = 0.1

    def _draw_pool(
        self, t: np.ndarray, *, m: int, seed: int
    ) -> list[np.ndarray] | None:
        return soft_pool_subsample_indices(
            t,
            K=self.subsample_samples,
            M=m,
            n_estimators=self.n_estimators,
            tau_half_frac=self.tau_half_frac,
            seed=seed,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SoftPoolContext":
        """Build from the shared pool knobs plus `tau_half_frac`."""
        return cls(
            tau_half_frac=float(config.get("tau_half_frac", 0.1)), **cls._shared(config)
        )
