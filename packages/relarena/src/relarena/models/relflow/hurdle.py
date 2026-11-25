"""Represent a zero-heavy regression target with numeric and null states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import relflow as rf


@dataclass(frozen=True)
class Hurdle:
    """A numeric target whose null state represents zero."""

    kind: str

    @classmethod
    def fit(cls, values: np.ndarray, *, target_kind: str) -> Hurdle | None:
        """Select from fitting labels with both positives and at least 5% zeros."""
        if target_kind not in {"ordinal", "number"}:
            raise ValueError(f"Unsupported hurdle target kind {target_kind!r}.")
        if not np.isfinite(values).all():
            raise ValueError("Regression fitting labels must be finite.")
        zeros = np.count_nonzero(values == 0)
        if zeros == len(values) or zeros < 0.05 * len(values) or (values < 0).any():
            return None
        return cls(target_kind)

    def postprocessor(self, model: rf.Model) -> rf.Postprocessor:
        """Bind the zero decision and positive readout to this fitted head."""
        # Both numeric heads use weighted state cross-entropy. Undo the fitted
        # relative weight in the null odds before applying the threshold.
        weights = model.nodes["/target"].embedder.counter.weight
        odds_scale = (weights[rf.Tokens.valued] / weights[rf.Tokens.null]).item()
        # Project a nonpositive readout onto the positive support. Valid
        # continuous predictions can fall below the smallest observed label.
        minimum = 1.0 if self.kind == "ordinal" else float(np.nextafter(0.0, 1.0))
        return positive_estimate.partial(odds_scale=odds_scale, minimum=minimum)


@rf.preprocess
def positive_target(frame: pl.DataFrame) -> pl.DataFrame:
    """Represent an observed zero as null while retaining positive labels."""
    values = frame["target"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("A hurdle target requires finite nonnegative labels.")
    return frame.with_columns(
        pl.when(pl.col("target") > 0)
        .then(pl.col("target"))
        .otherwise(None)
        .alias("target")
    )


def state_mass(
    target: pl.Series, odds_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the two admissible states and undo any relative null weighting."""
    state = target.struct.field("state")
    zero = state.struct.field("null").to_numpy().astype(np.float64) * odds_scale
    valued = state.struct.field("valued").to_numpy().astype(np.float64)
    mass = zero + valued
    if (
        not np.isfinite(mass).all()
        or (mass <= 0).any()
        or (zero < 0).any()
        or (valued < 0).any()
    ):
        raise ValueError(
            "The hurdle decoder must assign finite mass to zero or a positive amount."
        )
    return zero, valued


@rf.postprocess
def positive_estimate(
    frame: pl.DataFrame, *, odds_scale: float, minimum: float
) -> pl.DataFrame:
    """Use zero for a null decision and the native positive estimate otherwise."""
    target = frame["predictions"].struct.field("target")
    zero, valued = state_mass(target, odds_scale)
    p_zero = zero / (zero + valued)
    amounts = target.struct.field("content").to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(amounts).all():
        raise ValueError("The numeric hurdle returned a nonfinite amount.")
    prediction = np.where(p_zero >= 0.5, 0.0, np.maximum(minimum, amounts))
    return frame.select("inputs").with_columns(
        pl.Series("estimate", prediction), pl.Series("p_zero", p_zero)
    )
