"""Unit tests for the tabpfn-rel feature additions (`tabpfn_rel/features`).

Calendar / history-lag are pure-pandas and tested directly. The text path passes
raw anchor strings through (the estimator handles them; see
`models/_shared/tfm/tfm.py`), so the text tests exercise column detection, anchor
lookup, the raw pass-through, and the estimator overrides the pipeline emits.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table

from relarena.models.tabpfn_rel.features import (
    FeaturePipeline,
    RawTextFeaturizer,
    anchor_text_columns,
    attach_calendar,
    attach_history_lags,
    attach_text,
    build_history_pool,
)

TIME_COL = "timestamp"
ENTITY_COL = "user_id"
TARGET_COL = "label"


def _features(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": np.ones(n)})


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        entity_table="users",
        entity_col=ENTITY_COL,
        time_col=TIME_COL,
        target_col=TARGET_COL,
    )


# -- calendar ----------------------------------------------------------------


def test__attach_calendar__appends_sin_cos_pairs() -> None:
    raw = pd.DataFrame({TIME_COL: pd.to_datetime(["2020-01-15", "2020-07-01"])})
    out = attach_calendar(_features(2), raw, TIME_COL)
    expected_new = {
        "cutoff_month_sin",
        "cutoff_month_cos",
        "cutoff_dow_sin",
        "cutoff_dow_cos",
        "cutoff_doy_sin",
        "cutoff_doy_cos",
    }
    assert expected_new.issubset(out.columns)
    assert {"f0", "f1"}.issubset(out.columns)
    assert np.isclose(out["cutoff_month_sin"].iloc[0], np.sin(2 * np.pi * 1 / 12))
    assert (out[list(expected_new)].abs() <= 1.0 + 1e-9).all().all()


def test__attach_calendar__is_cyclic_wraparound_safe() -> None:
    raw = pd.DataFrame(
        {TIME_COL: pd.to_datetime(["2020-12-31", "2021-01-01", "2020-06-15"])}
    )
    out = attach_calendar(_features(3), raw, TIME_COL)
    pts = out[["cutoff_month_sin", "cutoff_month_cos"]].to_numpy()
    assert np.linalg.norm(pts[0] - pts[1]) < np.linalg.norm(pts[0] - pts[2])


def test__attach_calendar__noop_when_time_col_absent() -> None:
    feats = _features(3)
    raw = pd.DataFrame({"other": [1, 2, 3]})
    assert attach_calendar(feats, raw, None).equals(feats)
    assert attach_calendar(feats, raw, TIME_COL).equals(feats)


# -- history lags ------------------------------------------------------------


def _lags(
    df: pd.DataFrame, raw: pd.DataFrame, pool: pd.DataFrame | None, n_lags: int = 2
) -> pd.DataFrame:
    return attach_history_lags(
        df,
        raw,
        pool,
        time_col=TIME_COL,
        entity_col=ENTITY_COL,
        target_col=TARGET_COL,
        n_lags=n_lags,
    )


def test__attach_history_lags__most_recent_first_with_ages() -> None:
    pool = pd.DataFrame(
        {
            ENTITY_COL: ["u1", "u1", "u1"],
            TIME_COL: pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]),
            TARGET_COL: [10.0, 20.0, 30.0],
        }
    )
    raw = pd.DataFrame({ENTITY_COL: ["u1"], TIME_COL: pd.to_datetime(["2020-01-08"])})
    out = _lags(_features(1), raw, pool, n_lags=2)
    assert out["target_lag1"].iloc[0] == 30.0
    assert out["target_lag2"].iloc[0] == 20.0
    assert np.isclose(out["target_lag1_age_days"].iloc[0], 2.0)
    assert np.isclose(out["target_lag2_age_days"].iloc[0], 5.0)


def test__attach_history_lags__strict_lt_excludes_equal_timestamp() -> None:
    pool = pd.DataFrame(
        {
            ENTITY_COL: ["u1", "u1"],
            TIME_COL: pd.to_datetime(["2020-01-01", "2020-01-05"]),
            TARGET_COL: [10.0, 50.0],
        }
    )
    raw = pd.DataFrame({ENTITY_COL: ["u1"], TIME_COL: pd.to_datetime(["2020-01-05"])})
    out = _lags(_features(1), raw, pool, n_lags=1)
    assert out["target_lag1"].iloc[0] == 10.0  # not the equal-timestamp row


def test__attach_history_lags__nan_when_insufficient_history() -> None:
    pool = pd.DataFrame(
        {
            ENTITY_COL: ["u1"],
            TIME_COL: pd.to_datetime(["2020-01-01"]),
            TARGET_COL: [10.0],
        }
    )
    raw = pd.DataFrame(
        {
            ENTITY_COL: ["u1", "u2"],
            TIME_COL: pd.to_datetime(["2020-02-01", "2020-02-01"]),
        }
    )
    out = _lags(_features(2), raw, pool, n_lags=2)
    assert out["target_lag1"].iloc[0] == 10.0
    assert np.isnan(out["target_lag2"].iloc[0])
    assert np.isnan(out["target_lag1"].iloc[1])


def test__attach_history_lags__preserves_row_order_after_time_sort() -> None:
    pool = pd.DataFrame(
        {
            ENTITY_COL: ["u1", "u2"],
            TIME_COL: pd.to_datetime(["2019-01-01", "2019-06-01"]),
            TARGET_COL: [1.0, 2.0],
        }
    )
    raw = pd.DataFrame(
        {
            ENTITY_COL: ["u2", "u1"],
            TIME_COL: pd.to_datetime(["2020-01-01", "2020-01-01"]),
        }
    )
    df = _features(2)
    df["marker"] = [100.0, 200.0]
    out = _lags(df, raw, pool, n_lags=1)
    assert out["marker"].tolist() == [100.0, 200.0]
    assert out["target_lag1"].iloc[0] == 2.0  # u2
    assert out["target_lag1"].iloc[1] == 1.0  # u1


def test__attach_history_lags__noop_when_disabled() -> None:
    feats = _features(2)
    raw = pd.DataFrame(
        {ENTITY_COL: ["u1", "u2"], TIME_COL: pd.to_datetime(["2020-01-01"] * 2)}
    )
    assert _lags(feats, raw, None, n_lags=2).equals(feats)
    assert _lags(feats, raw, _features(0), n_lags=0).equals(feats)


def test__build_history_pool__selects_three_columns() -> None:
    label_df = pd.DataFrame(
        {
            ENTITY_COL: ["u1"],
            TIME_COL: pd.to_datetime(["2020-01-01"]),
            TARGET_COL: [1.0],
            "extra": ["ignored"],
        }
    )
    pool = build_history_pool(
        label_df, entity_col=ENTITY_COL, time_col=TIME_COL, target_col=TARGET_COL
    )
    assert list(pool.columns) == [ENTITY_COL, TIME_COL, TARGET_COL]


def test__attach_text__concat_and_alignment() -> None:
    feats = pd.DataFrame({"f0": [1.0, 2.0]})
    out = attach_text(feats, pd.DataFrame({"e0": [9, 8]}))
    assert list(out.columns) == ["f0", "e0"] and out["e0"].tolist() == [9, 8]
    assert attach_text(feats, None).equals(feats)
    with pytest.raises(ValueError, match="row mismatch"):
        attach_text(feats, pd.DataFrame({"e0": [1, 2, 3]}))


# -- text features -----------------------------------------------------------


def _db() -> SimpleNamespace:
    users = SimpleNamespace(
        df=pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "bio": ["hello world", "hi", "a longer biography here"],
                "city": ["NYC", "LA", "SF"],
                "age": [30, 40, 50],
                "signup": pd.to_datetime(["2019-01-01"] * 3),
            }
        ),
        pkey_col="user_id",
        time_col="signup",
        fkey_col_to_pkey_table={},
    )
    return SimpleNamespace(table_dict={"users": users})


def _split_table(user_ids: list[int]) -> Table:
    return Table(
        df=pd.DataFrame(
            {
                ENTITY_COL: user_ids,
                TIME_COL: pd.to_datetime(["2020-01-01"] * len(user_ids)),
                TARGET_COL: [0.0] * len(user_ids),
            }
        ),
        fkey_col_to_pkey_table={ENTITY_COL: "users"},
        pkey_col=None,
        time_col=TIME_COL,
    )


def test__anchor_text_columns__picks_object_cols_excluding_keys_time() -> None:
    assert set(anchor_text_columns(_db(), _task())) == {"bio", "city"}


def test__RawTextFeaturizer__fit_returns_raw_text_aligned_by_key() -> None:
    db, task = _db(), _task()
    train = _split_table([3, 1])  # row0=user3, row1=user1
    raw = RawTextFeaturizer().fit(db, task, train)
    assert raw is not None and len(raw) == 2
    assert raw["bio__raw_text"].tolist() == ["a longer biography here", "hello world"]
    assert raw["city__raw_text"].tolist() == ["SF", "NYC"]


def test__RawTextFeaturizer__none_when_no_text_columns() -> None:
    db, task = _db(), _task()
    db.table_dict["users"].df = db.table_dict["users"].df[["user_id", "age", "signup"]]
    feat = RawTextFeaturizer()
    assert feat.fit(db, task, _split_table([1, 2, 3])) is None
    assert feat.transform(db, task, _split_table([1])) is None


def test__RawTextFeaturizer__transform_uses_columns_detected_at_fit() -> None:
    db, task = _db(), _task()
    feat = RawTextFeaturizer()
    feat.fit(db, task, _split_table([1, 2, 3]))
    out = feat.transform(db, task, _split_table([2, 3]))
    assert out is not None and out["bio__raw_text"].tolist() == [
        "hi",
        "a longer biography here",
    ]


def test__FeaturePipeline__no_text_columns_when_anchor_has_no_text() -> None:
    # with_text on, but the anchor table carries no text columns: the fit stays
    # text-free.
    db, task = _db(), _task()
    db.table_dict["users"].df = db.table_dict["users"].df.drop(columns=["bio", "city"])
    pipe = FeaturePipeline({"with_text": True})
    df = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    out = pipe.fit_transform(df, task, _split_table([1, 2, 3]), db)
    assert list(out.columns) == ["f"]


# -- FeaturePipeline ---------------------------------------------------------


def test__FeaturePipeline__no_knobs__returns_frame_unchanged() -> None:
    pipe = FeaturePipeline({})
    assert not pipe.enabled
    feats = _features(3)
    train = _split_table([1, 2, 3])
    assert pipe.fit_transform(feats, _task(), train, _db()).equals(feats)


def test__FeaturePipeline__applies_calendar_history_text_in_order() -> None:
    pipe = FeaturePipeline(
        {
            "with_calendar_features": True,
            "with_history_features": True,
            "n_lags": 2,
            "with_text": True,
        }
    )
    train = _split_table([1, 2, 3])
    out = pipe.fit_transform(_features(3), _task(), train, _db())
    assert "cutoff_month_sin" in out.columns  # calendar
    assert "target_lag1" in out.columns  # history lags
    assert "bio__raw_text" in out.columns  # text


def test__FeaturePipeline__transform_reuses_fit_state() -> None:
    pipe = FeaturePipeline({"with_history_features": True, "n_lags": 1})
    train = _split_table([1, 1, 2])
    train.df[TIME_COL] = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-01"])
    train.df[TARGET_COL] = [10.0, 20.0, 5.0]
    pipe.fit_transform(_features(3), _task(), train, _db())
    val = _split_table([1])
    val.df[TIME_COL] = pd.to_datetime(["2020-01-10"])
    out = pipe.transform(_features(1), _task(), val, _db())
    assert out["target_lag1"].iloc[0] == 20.0  # most recent prior train label for u1
