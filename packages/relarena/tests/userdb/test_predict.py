"""Tests for production-inference seed construction."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from relbench.base import Database, Table

from relarena.userdb.predict import make_prediction_table


def test__make_prediction_table__explicit_ids__drops_absent_entities() -> None:
    """Explicit ids absent at the anchor are dropped, with a warning."""
    drivers = Table(
        df=pd.DataFrame(
            {
                "driver_id": [0, 1, 2],
                "created": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-03-01"]),
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col="driver_id",
        time_col="created",
    )
    db = Database({"drivers": drivers})
    task = SimpleNamespace(entity_table="drivers", entity_col="driver_id", time_col="t")

    # driver 2 (created 2020-03-01) does not exist at the anchor -> dropped + warned.
    with pytest.warns(UserWarning, match="Dropped 1 requested entity id"):
        seed = make_prediction_table(
            db, task, pd.Timestamp("2020-02-01"), entities=[0, 1, 2]
        )

    assert sorted(seed.df["driver_id"].tolist()) == [0, 1]


def test__make_prediction_table__explicit_ids_static_table__drops_unknown() -> None:
    """Explicit ids are validated even for a static (non-temporal) entity table."""
    drivers = Table(
        df=pd.DataFrame({"driver_id": [0, 1]}),
        fkey_col_to_pkey_table={},
        pkey_col="driver_id",
        time_col=None,
    )
    db = Database({"drivers": drivers})
    task = SimpleNamespace(entity_table="drivers", entity_col="driver_id", time_col="t")

    # 999 is not in the static entity table -> dropped + warned (no time_col needed).
    with pytest.warns(UserWarning, match="Dropped 1 requested entity id"):
        seed = make_prediction_table(
            db, task, pd.Timestamp("2020-02-01"), entities=[0, 999]
        )

    assert sorted(seed.df["driver_id"].tolist()) == [0]


def test__make_prediction_table__scalar_entities__raises() -> None:
    """A bare scalar (not a list) is rejected with a clear error, not a TypeError."""
    drivers = Table(
        df=pd.DataFrame({"driver_id": [0, 1, 2]}),
        fkey_col_to_pkey_table={},
        pkey_col="driver_id",
        time_col=None,
    )
    db = Database({"drivers": drivers})
    task = SimpleNamespace(entity_table="drivers", entity_col="driver_id", time_col="t")

    with pytest.raises(ValueError, match="got scalar"):
        make_prediction_table(db, task, pd.Timestamp("2020-02-01"), entities=5)
