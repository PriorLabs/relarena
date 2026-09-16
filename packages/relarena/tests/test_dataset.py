"""Tests for split construction (no data download — real in-memory RelBench objects).

These pin the temporal-correctness invariant: the inner split carries a DB
censored at `val_timestamp` and scores against the val table; the outer split
carries the (test-censored) DB, trains on the train+val union, and leaves the
eval target hidden so RelBench supplies the test labels.

The stand-in database is a *real* `relbench.base.Database` built from tiny
in-memory tables (a timestamped fact table plus a static dimension table), so the
censoring is exercised through RelBench's actual `Database.upto` across multiple
tables — not a fake `.upto` — while still needing no download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from relbench.base import Database, Dataset, Table

from relarena.dataset import (
    InnerSplit,
    OuterSplit,
    RelBenchDatasetTask,
    drop_noncanonical_columns,
)

#: Five monthly event timestamps; cut the inner split after the 3rd, the outer
#: (test) split after the 5th — so censoring at val should drop the last two rows.
EVENT_TIMES = pd.to_datetime(
    ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
)
VAL_TS = pd.Timestamp("2020-03-01")
TEST_TS = pd.Timestamp("2020-05-01")


def _db() -> Database:
    """A real multi-table RelBench `Database` with no download.

    Two tables exercise both censoring paths: `events` is timestamped (rows past the
    cutoff are dropped) and `drivers` is a static dimension with `time_col=None`
    (passes `upto` untouched). Pkeys are 0-indexed, as RelBench's
    `reindex_pkeys_and_fkeys` leaves them — so `validate_and_correct_db` (which
    asserts consecutive 0..N-1 pkeys) accepts the censored DB.
    """
    events = Table(
        df=pd.DataFrame(
            {
                "event_id": [0, 1, 2, 3, 4],
                "driver_id": [0, 1, 2, 0, 1],
                "t": EVENT_TIMES,
            }
        ),
        fkey_col_to_pkey_table={"driver_id": "drivers"},
        pkey_col="event_id",
        time_col="t",
    )
    drivers = Table(
        df=pd.DataFrame({"driver_id": [0, 1, 2], "name": ["a", "b", "c"]}),
        fkey_col_to_pkey_table={},
        pkey_col="driver_id",
        time_col=None,
    )
    return Database({"events": events, "drivers": drivers})


def _table(entities: list[int], times: list[int], ys: list[float]) -> Table:
    """A task label table (train/val/test), separate from the database tables."""
    return Table(
        df=pd.DataFrame({"entity": entities, "t": times, "y": ys}),
        fkey_col_to_pkey_table={"entity": "drivers"},
        pkey_col=None,
        time_col="t",
    )


def _source() -> RelBenchDatasetTask:
    """A source with stand-in internals, bypassing the (downloading) __init__.

    The task entity is the static `drivers` dimension; its seeds (0..2) are all in
    range, so inner_split's FK/seed scrubs are no-ops here — this fixture exercises
    only the baseline censoring. The dangling-FK / dangling-seed regressions use the
    separate `_temporal_entity_source`.
    """
    src = object.__new__(RelBenchDatasetTask)
    src.dataset_name = "ds"
    src.task_name = "task"
    # Real RelBench corrector (its body ignores `self`), so inner_split scrubs
    # dangling FKs exactly as get_db does for the test cutoff.
    src._dataset = SimpleNamespace(
        val_timestamp=VAL_TS,
        test_timestamp=TEST_TS,
        validate_and_correct_db=lambda db: Dataset.validate_and_correct_db(None, db),
    )
    src._db = _db()
    src._task = SimpleNamespace(entity_table="drivers", entity_col="entity")
    src._tables = {
        "train": _table([1, 2], [10, 11], [0.0, 1.0]),
        "val": _table([2], [12], [2.0]),
        "test": _table([0, 1], [13, 14], [3.0, 4.0]),
    }
    return src


def _temporal_entity_source() -> RelBenchDatasetTask:
    """A source whose *entity* table is timestamped, so the val cutoff shrinks it.

    Kept separate from `_source` so the baseline split tests are not perturbed.
    The entity table `events` straddles the val cutoff (events 3,4 are dropped);
    `attendance` carries a forward-pointing FK (an early row referencing event 4),
    and the seed tables reference `events` — the two setups the dangling-FK and
    dangling-seed regressions need.
    """

    def _event_table(entities: list[int]) -> Table:
        return Table(
            df=pd.DataFrame({"entity": entities, "t": [12] * len(entities)}),
            fkey_col_to_pkey_table={"entity": "events"},
            pkey_col=None,
            time_col="t",
        )

    events = Table(
        df=pd.DataFrame({"event_id": [0, 1, 2, 3, 4], "t": EVENT_TIMES}),
        fkey_col_to_pkey_table={},
        pkey_col="event_id",
        time_col="t",
    )
    # An early (kept) attendance row references event 4 (May), dropped at the val
    # cutoff — a forward-in-time FK that dangles once that event is gone.
    attendance = Table(
        df=pd.DataFrame(
            {
                "attendance_id": [0, 1, 2],
                "event_id": [0, 1, 4],
                "t": pd.to_datetime(["2020-01-10", "2020-02-10", "2020-01-20"]),
            }
        ),
        fkey_col_to_pkey_table={"event_id": "events"},
        pkey_col="attendance_id",
        time_col="t",
    )
    src = object.__new__(RelBenchDatasetTask)
    src.dataset_name = "ds"
    src.task_name = "task"
    src._dataset = SimpleNamespace(
        val_timestamp=VAL_TS,
        test_timestamp=TEST_TS,
        validate_and_correct_db=lambda db: Dataset.validate_and_correct_db(None, db),
    )
    src._db = Database({"events": events, "attendance": attendance})
    src._task = SimpleNamespace(entity_table="events", entity_col="entity")
    src._tables = {
        "train": _event_table([0, 1]),
        "val": _event_table([2]),
        "test": _event_table([3, 4]),
    }
    return src


def _noisy_db() -> Database:
    """A DB exercising each dataset-agnostic drop rule plus a column to keep.

    `facts` carries an `Unnamed: N` CSV row-index artifact and a fully-NaN
    column (both dropped) alongside a sparse-but-populated column (kept); `clean`
    has nothing to drop, so it must survive as the same object.
    """
    facts = Table(
        df=pd.DataFrame(
            {
                "Unnamed: 0": [0, 1],
                "driver_id": [1, 2],
                "all_nan": [None, None],
                "sparse": ["x", None],
            }
        ),
        fkey_col_to_pkey_table={"driver_id": "clean"},
        pkey_col=None,
        time_col=None,
    )
    clean = Table(
        df=pd.DataFrame({"driver_id": [1, 2], "name": ["a", "b"]}),
        fkey_col_to_pkey_table={},
        pkey_col="driver_id",
        time_col=None,
    )
    return Database({"facts": facts, "clean": clean})


def _ratebeer_users_db(*, with_aggregate: bool = True) -> Database:
    """A one-table DB standing in for rel-ratebeer's `users`.

    Carries a time-leaking full-history aggregate (unless disabled) that is dropped
    only when the dataset is rel-ratebeer.
    """
    cols: dict[str, list[object]] = {"user_id": [1, 2], "name": ["a", "b"]}
    if with_aggregate:
        cols["max_beer_rating"] = [4.5, 3.0]
    users = Table(
        df=pd.DataFrame(cols),
        fkey_col_to_pkey_table={},
        pkey_col="user_id",
        time_col=None,
    )
    return Database({"users": users})


def test__drop_noncanonical_columns__artifacts_and_nan__dropped_sparse_kept() -> None:
    db = _noisy_db()
    out = drop_noncanonical_columns(db, "rel-event")

    assert list(out.table_dict["facts"].df.columns) == ["driver_id", "sparse"]
    # Untouched tables keep their identity.
    assert out.table_dict["clean"] is db.table_dict["clean"]
    # Input not mutated: checksum verification reads relbench's lru_cached raw db.
    assert "Unnamed: 0" in db.table_dict["facts"].df.columns


def test__drop_noncanonical_columns__rebuilt_table__preserves_schema() -> None:
    db = _noisy_db()
    rebuilt, original = (
        drop_noncanonical_columns(db, "rel-event").table_dict["facts"],
        db.table_dict["facts"],
    )
    assert rebuilt.fkey_col_to_pkey_table == original.fkey_col_to_pkey_table
    assert rebuilt.pkey_col == original.pkey_col
    assert rebuilt.time_col == original.time_col
    assert list(rebuilt.df["driver_id"]) == [1, 2]


def test__drop_noncanonical_columns__ratebeer_user_aggregate__dropped() -> None:
    out = drop_noncanonical_columns(_ratebeer_users_db(), "rel-ratebeer")
    assert list(out.table_dict["users"].df.columns) == ["user_id", "name"]


def test__drop_noncanonical_columns__aggregate_name_other_dataset__kept() -> None:
    # The aggregate drop is scoped to rel-ratebeer; the same name elsewhere is a
    # real feature and stays.
    db = _ratebeer_users_db()
    assert drop_noncanonical_columns(db, "rel-stack") is db


def test__drop_noncanonical_columns__clean_db__returns_input_unchanged() -> None:
    # Nothing to drop -> the input object is returned as-is.
    db = _ratebeer_users_db(with_aggregate=False)
    assert drop_noncanonical_columns(db, "rel-ratebeer") is db


def test_inner_split_censors_db_at_val_and_scores_on_val() -> None:
    src = _source()
    inner = src.inner_split()

    assert isinstance(inner, InnerSplit)
    assert inner.name == "inner"
    assert inner.cutoff == VAL_TS

    # The DB was censored at the val cutoff, across all tables: the timestamped
    # `events` table keeps only rows up to VAL_TS (3 of 5), while the static
    # `drivers` dimension (time_col=None) passes through whole.
    assert list(inner.db_state.table_dict["events"].df["t"]) == list(EVENT_TIMES[:3])
    assert (inner.db_state.table_dict["events"].df["t"] <= VAL_TS).all()
    assert len(inner.db_state.table_dict["drivers"].df) == 3
    # Censoring returns a new DB; the source's DB is left uncensored.
    assert len(src._db.table_dict["events"].df) == 5

    assert inner.train_table is src._tables["train"]
    assert inner.eval_table is src._tables["val"]
    assert inner.eval_target is src._tables["val"]  # val labels are not hidden


def test__inner_split__forward_pointing_fkey__scrubbed_not_dangling() -> None:
    # Regression: the val-cutoff drops the May event, orphaning the early attendance
    # row that referenced it. inner_split must re-run validate_and_correct_db so the
    # dangling FK is nulled — otherwise make_pkey_fkey_graph asserts on the
    # out-of-range index (the failure seen on rel-event / rel-stack).
    inner = _temporal_entity_source().inner_split()
    attendance = inner.db_state.table_dict["attendance"].df
    n_events = len(inner.db_state.table_dict["events"].df)  # 3 survive the val cutoff

    fkeys = attendance["event_id"]
    assert fkeys.isna().sum() == 1  # the forward FK (event 4) was scrubbed to null
    assert (fkeys.dropna() < n_events).all()  # no surviving FK is out of range


def test__inner_split__shared_timeless_table__does_not_corrupt_outer_split() -> None:
    # Regression, exercised through the real inner_split -> outer_split flow the
    # runner uses (both called on one source). `Table.upto` returns timeless tables
    # (`time_col=None`) as the *same object* held in `src._db`, and the real
    # `validate_and_correct_db` scrubs dangling FKs in place; without copying those
    # shared frames the scrub leaks into `src._db` and corrupts the DB outer_split
    # later hands the model. A timeless `badges` table forward-references the May
    # event, which the val cutoff censors out but the test cutoff keeps.
    src = _source()
    src._db.table_dict["badges"] = Table(
        df=pd.DataFrame({"badge_id": [0, 1], "event_id": [0, 4]}),
        fkey_col_to_pkey_table={"event_id": "events"},
        pkey_col="badge_id",
        time_col=None,
    )

    inner = src.inner_split()
    outer = src.outer_split()

    # inner_split scrubbed its own (private) copy: the forward FK to the censored
    # May event is nulled.
    assert inner.db_state.table_dict["badges"].df["event_id"].isna().sum() == 1
    # outer_split (test cutoff) keeps event 4, so its FK is valid and must survive
    # intact -- it would read `[0, <NA>]` if the inner scrub had leaked through.
    assert outer.db_state.table_dict["badges"].df["event_id"].tolist() == [0, 4]


def test__inner_split__seed_for_post_val_entity__dropped() -> None:
    # Regression: a seed referencing event 4 (May) — censored out at val_timestamp —
    # must be dropped, else the graph sampler indexes past the shrunken entity table.
    # relbench's get_table only filtered seeds against the test count, so this is the
    # val-cutoff analog of filter_dangling_entities.
    src = _temporal_entity_source()
    src._tables["val"] = Table(
        df=pd.DataFrame({"entity": [2, 4], "t": [12, 12]}),  # event 2 in, event 4 out
        fkey_col_to_pkey_table={"entity": "events"},
        pkey_col=None,
        time_col="t",
    )
    inner = src.inner_split()

    assert inner.eval_table.df["entity"].tolist() == [2]  # post-val seed dropped
    assert inner.eval_table is inner.eval_target  # filtered once; pred/target aligned


def test_outer_split_exposes_train_val_separately_and_hides_test_labels() -> None:
    src = _source()
    outer = src.outer_split()

    assert isinstance(outer, OuterSplit)
    assert outer.name == "outer"
    assert outer.cutoff == TEST_TS
    # The DB is censored at test_timestamp. Since get_db() already censored at
    # test_timestamp, this re-censor is a no-op: every event row is retained.
    assert (outer.db_state.table_dict["events"].df["t"] <= TEST_TS).all()
    assert len(outer.db_state.table_dict["events"].df) == 5
    # train and val are exposed separately (not pre-unioned), schema preserved.
    assert outer.train_table is src._tables["train"]
    assert list(outer.train_table.df["y"]) == [0.0, 1.0]
    assert outer.train_table.time_col == "t"
    assert outer.val_table is src._tables["val"]
    assert list(outer.val_table.df["y"]) == [2.0]
    assert outer.eval_table is src._tables["test"]
    # No eval target on the outer split: the test labels are never materialized
    # here — scoring passes target_table=None and RelBench supplies them.
    assert not hasattr(outer, "eval_target")
