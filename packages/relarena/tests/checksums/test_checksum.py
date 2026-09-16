"""Unit tests for the content-checksum primitives (no data download).

Pins the frozen `array_checksum` values (parity with the benchmarking package)
and the table/database invariants that matter for RelBench data — including the
`list`-valued and pandas-nullable columns that break a naive
`hash_pandas_object`. The end-to-end split checksums live in
`relarena.checksums`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Dataset, Table

import relarena.checksums.checksum as ck
from relarena.checksums import array_checksum, database_checksum, table_checksum
from relarena.dataset import RelBenchDatasetTask


def _table(df: pd.DataFrame, **kw) -> Table:
    kw.setdefault("fkey_col_to_pkey_table", {})
    kw.setdefault("pkey_col", "id")
    kw.setdefault("time_col", "t")
    return Table(df=df, **kw)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [1.5, 2.5, np.nan],
            "name": ["x", "y", "z"],
            "ok": [True, False, True],
            "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )


# -- array_checksum --------------------------------------------------------------


@pytest.mark.parametrize(
    "arr, expected",
    [
        (np.array([1, 2, 3, 4], dtype=np.int32), 41),  # frozen: parity w/ benchmarking
        (np.arange(16, dtype=np.uint8), 255),
        (np.array([], dtype=np.int32), 0),  # empty -> 0
    ],
)
def test_array_checksum_frozen_values(arr: np.ndarray, expected: int) -> None:
    assert array_checksum(arr) == np.uint64(expected)


def test_array_checksum_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError):
        array_checksum(np.array([1 + 2j], dtype=np.complex128))


# -- table_checksum: determinism + value/schema/dtype sensitivity ----------------


def test_table_checksum_is_deterministic_and_value_sensitive() -> None:
    base = table_checksum(_table(_df()))
    assert table_checksum(_table(_df())) == base
    changed = _df()
    changed.loc[0, "name"] = "X"
    assert table_checksum(_table(changed)) != base


def test_table_checksum_detects_schema_change() -> None:
    base = table_checksum(_table(_df()))
    assert table_checksum(_table(_df(), time_col=None)) != base
    assert table_checksum(_table(_df(), fkey_col_to_pkey_table={"id": "u"})) != base


@pytest.mark.parametrize(
    "df, idx, col, new_value",
    [
        # list<string> columns (amazon's product.category) load as object cells
        # holding ndarrays — the case that crashes pd.util.hash_pandas_object.
        (
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "cat": [np.array(["a", "b"]), np.array(["c"]), np.array([])],
                }
            ),
            0,
            "cat",
            np.array(["a", "B"]),
        ),
        # pandas nullable Int64 with NA -> object array under to_numpy(); must stay
        # deterministic (the pointer-view bug) and sensitive to the NA cell's value.
        (
            pd.DataFrame(
                {"id": [1, 2, 3, 4], "v": pd.array([1, 2, pd.NA, 4], "Int64")}
            ),
            2,
            "v",
            99,
        ),
    ],
)
def test_table_checksum_handles_tricky_dtypes(
    df: pd.DataFrame, idx: int, col: str, new_value: object
) -> None:
    base = table_checksum(_table(df, time_col=None))
    assert table_checksum(_table(df.copy(), time_col=None)) == base  # deterministic
    changed = df.copy()
    changed.at[idx, col] = new_value
    assert table_checksum(_table(changed, time_col=None)) != base  # value-sensitive


# -- database_checksum -----------------------------------------------------------


def test_database_checksum_is_name_and_content_sensitive() -> None:
    a, b = _table(_df()), _table(_df().assign(price=[9.0, 8.0, 7.0]))
    base = database_checksum(Database({"a": a, "b": b}))
    assert database_checksum(Database({"a": a, "b": b})) == base
    # swapping which name maps to which table must change the result
    assert database_checksum(Database({"a": b, "b": a})) != base
    # and so must changing a table's content
    changed = _df()
    changed.loc[1, "id"] = 99
    assert database_checksum(Database({"a": a, "b": _table(changed)})) != base


# -- record_checksums / check_checksums (compute mocked; no data download) --------


def _patch_compute(
    monkeypatch: pytest.MonkeyPatch, computed: dict[str, dict[str, int]]
) -> None:
    """Stub the heavy, download-bound compute so we can test compare/record logic."""
    monkeypatch.setattr(
        ck, "_iter_checksums", lambda specs, *, download=True: iter(computed.items())
    )


def test_record_checksums_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"d/t": {"train": 1}}))
    _patch_compute(monkeypatch, {"d/t": {"train": 2}})
    ck.record_checksums([("d", "t")], path)
    assert json.loads(path.read_text())["d/t"]["train"] == 2


def test_check_checksums_reports_mismatches_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.json"
    original = json.dumps({"d/t": {"train": 1, "test": 2}})
    path.write_text(original)
    # one changed split, one unchanged split, and a task absent from the baseline
    _patch_compute(monkeypatch, {"d/t": {"train": 1, "test": 999}, "d/u": {"train": 5}})
    mism = ck.check_checksums([("d", "t"), ("d", "u")], path)
    assert mism["d/t"]["test"] == (2, 999)
    assert "train" not in mism["d/t"]  # unchanged split not reported
    assert mism["d/u"]["train"] == (None, 5)  # missing-from-baseline task
    assert path.read_text() == original  # baseline untouched

    # matching compute -> no mismatches
    _patch_compute(monkeypatch, {"d/t": {"train": 1, "test": 2}})
    assert ck.check_checksums([("d", "t")], path) == {}


# -- _source_checksums: hashes the model-facing splits + hidden test labels -------


class _FakeTask:
    """Minimal EntityTask stand-in exercising RelBench's masking semantics.

    `mask_input_cols=True` drops the target (as RelBench does to hide the test
    answer key); `test_labels` is recorded with `mask_input_cols=False`, so
    the target must be covered.
    """

    target_col = "y"
    entity_table = "users"
    entity_col = "user"
    time_col = "t"

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables

    def get_table(self, split: str, mask_input_cols: bool = False) -> Table:
        df = self._tables[split]
        if mask_input_cols:
            df = df.drop(columns=[self.target_col])
        return Table(
            df=df, fkey_col_to_pkey_table={"user": "users"}, pkey_col=None, time_col="t"
        )


def _fake_source(test_label: float = 1.0) -> RelBenchDatasetTask:
    """A source with stand-in internals, bypassing the (downloading) __init__.

    The db's `events` table straddles the val cutoff so the inner (val-censored)
    and outer (test-censored) databases genuinely differ.
    """
    times = pd.to_datetime(["2020-01-01", "2020-06-01"])
    events = Table(
        df=pd.DataFrame({"eid": [0, 1], "t": times}),
        fkey_col_to_pkey_table={},
        pkey_col="eid",
        time_col="t",
    )
    # Static entity table the seed "user" ids index into (so inner_split's
    # dangling-seed filter has a table to size against; all seeds 1,2 are in range).
    users = Table(
        df=pd.DataFrame({"user": [0, 1, 2]}),
        fkey_col_to_pkey_table={},
        pkey_col="user",
        time_col=None,
    )
    labels = {
        split: pd.DataFrame({"user": [1, 2], "t": [10, 11], "y": [0.0, 1.0]})
        for split in ("train", "val")
    }
    labels["test"] = pd.DataFrame(
        {"user": [1, 2], "t": [10, 11], "y": [0.0, test_label]}
    )
    task = _FakeTask(labels)
    src = object.__new__(RelBenchDatasetTask)
    src.dataset_name = "ds"
    src.task_name = "task"
    src._dataset = SimpleNamespace(
        val_timestamp=pd.Timestamp("2020-03-01"),
        test_timestamp=pd.Timestamp("2020-06-01"),
        validate_and_correct_db=lambda db: Dataset.validate_and_correct_db(None, db),
    )
    src._db = Database({"events": events, "users": users})
    src._task = task
    src._tables = {
        "train": task.get_table("train"),
        "val": task.get_table("val"),
        "test": task.get_table("test", mask_input_cols=True),  # masked, as RelBench
    }
    return src


def test__db_and_label_checksums__cover_splits_and_unmasked_test_labels() -> None:
    src = _fake_source()
    cs = {**ck._db_checksums(src), **ck._label_checksums(src)}

    assert sorted(cs) == [
        "inner_db",
        "inner_eval",
        "inner_train",
        "outer_db",
        "outer_eval",
        "outer_train",
        "outer_val",
        "test_labels",
    ]
    inner, outer = src.inner_split(), src.outer_split()
    assert cs["inner_db"] == int(ck.database_checksum(inner.db_state))
    assert cs["inner_db"] != cs["outer_db"]  # val censoring drops a row
    # Outer train and val are fingerprinted independently (not pre-unioned).
    assert cs["outer_train"] == int(table_checksum(outer.train_table))
    assert cs["outer_val"] == int(table_checksum(outer.val_table))
    assert cs["outer_eval"] == int(table_checksum(outer.eval_table))
    # outer_eval is the *masked* test table; the answer key is pinned separately.
    assert cs["test_labels"] != cs["outer_eval"]
    assert cs["test_labels"] == int(
        table_checksum(src._task.get_table("test", mask_input_cols=False))
    )


def test__label_checksums__test_label_change__caught_and_isolated() -> None:
    base = ck._label_checksums(_fake_source())
    after = ck._label_checksums(_fake_source(test_label=99.0))
    # only the hidden-label checksum moves: the masked outer_eval can't see y, and
    # the train/val label tables don't carry the flipped value.
    assert after["test_labels"] != base["test_labels"]
    assert {k: v for k, v in after.items() if k != "test_labels"} == {
        k: v for k, v in base.items() if k != "test_labels"
    }


def test__iter_checksums__hashes_each_dataset_db_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ck, "RelBenchDatasetTask", lambda name, task, download: _fake_source()
    )
    calls = {"n": 0}
    real = ck._db_checksums

    def counting(source: RelBenchDatasetTask) -> dict[str, int]:
        calls["n"] += 1
        return real(source)

    monkeypatch.setattr(ck, "_db_checksums", counting)
    out = dict(ck._iter_checksums([("ds", "t1"), ("ds", "t2"), ("ds2", "t1")]))
    assert calls["n"] == 2  # once per dataset, not per task
    assert sorted(out) == ["ds/t1", "ds/t2", "ds2/t1"]
