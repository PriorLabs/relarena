"""Unit tests for the DFS depth cache and RDB construction (no data download)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

import relarena.featurization.dfs as dfs_mod
from relarena.cache import CacheConfig
from relarena.featurization.dfs import (
    TARGET_HISTORY_TABLE_NAME,
    _DepthCache,
    _temporal_diff,
)
from relarena.identity import RunIdentity


def _toy_db() -> Database:
    """A tiny relbench `Database`: keyed `users` + a keyless `reviews` child."""
    users = Table(
        df=pd.DataFrame(
            {
                "uid": [1, 2, 3],
                "age": [20.0, 30.0, 40.0],
                "signup": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col="uid",
        time_col=None,
    )
    cols: dict = {
        "uid": [1, 2, 1, 3],
        "rating": [5.0, 4.0, 3.0, 2.0],
        "ts": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
    }
    reviews = Table(
        df=pd.DataFrame(cols),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,  # keyless fact table
        time_col="ts",
    )
    return Database({"users": users, "reviews": reviews})


def test_depth_cache_memoizes_matrix_and_depth_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dfs_mod, "_build_rdb", lambda db: f"rdb::{id(db)}")
    cache = _DepthCache()
    db = object()
    assert cache.raw_rdb_for(db) == f"rdb::{id(db)}"

    df = pd.DataFrame({"a": [1, 2]})
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"f": [1, 2]})

    config = CacheConfig(None, "compute")
    cache.full_matrix(db, df, None, 2, config, compute)
    cache.full_matrix(db, df, None, 2, config, compute)
    assert calls["n"] == 1
    assert cache.matrix_computations == 1

    cache.full_matrix(db, pd.DataFrame({"a": [9]}), None, 2, config, compute)
    assert calls["n"] == 2
    assert cache.matrix_computations == 2

    hist = pd.DataFrame({"y": [1]})
    cache.full_matrix(db, df, hist, 2, config, compute)
    assert calls["n"] == 3
    cache.full_matrix(db, df, hist, 2, config, compute)
    assert calls["n"] == 3

    dm_calls = {"n": 0}

    def depths() -> dict:
        dm_calls["n"] += 1
        return {"drivers.COUNT(x)": 2}

    cache.depth_map(None, 2, config, depths)
    cache.depth_map(None, 2, config, depths)
    assert dm_calls["n"] == 1
    cache.depth_map(hist, 2, config, depths)
    assert dm_calls["n"] == 2


def test_depth_cache_variant_builds_once_and_holds_history_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dfs_mod, "_build_rdb", lambda db: "raw")
    cache = _DepthCache()
    db = object()
    builds = {"n": 0}

    def build(raw: str) -> str:
        builds["n"] += 1
        return f"variant::{raw}"

    hist = pd.DataFrame({"y": [1]})
    assert cache.rdb_variant(db, hist, build) == "variant::raw"
    assert cache.rdb_variant(db, hist, build) == "variant::raw"  # memoized
    assert builds["n"] == 1
    cache.rdb_variant(db, None, build)  # base variant is a separate slot
    assert builds["n"] == 2
    # the history frame reference is held, so its id can't be reused after GC
    assert cache._rdbs[id(hist)][0] is hist


def test__depth_cache__different_policy__does_not_hide_a_fill(tmp_path: Path) -> None:
    cache = _DepthCache()
    db = object()
    source = pd.DataFrame({"a": [1]})
    calls = {"matrix": 0, "depths": 0}

    def matrix() -> pd.DataFrame:
        calls["matrix"] += 1
        return pd.DataFrame({"f": [calls["matrix"]]})

    def depths() -> dict[str, int]:
        calls["depths"] += 1
        return {"f": calls["depths"]}

    compute_config = CacheConfig(tmp_path, "compute")
    fill_config = CacheConfig(tmp_path, "fill")
    computed = cache.full_matrix(db, source, None, 2, compute_config, matrix)
    assert computed["f"].tolist() == [1]
    assert cache.depth_map(None, 2, compute_config, depths) == {"f": 1}
    filled = cache.full_matrix(db, source, None, 2, fill_config, matrix)
    assert filled["f"].tolist() == [2]
    assert cache.depth_map(None, 2, fill_config, depths) == {"f": 2}


def test__depth_cache__different_db_without_rdb_build__does_not_reuse_matrix() -> None:
    cache = _DepthCache()
    source = pd.DataFrame({"a": [1]})
    history = pd.DataFrame({"y": [1]})
    config = CacheConfig(None, "compute")
    inner_db = object()
    outer_db = object()

    first = cache.full_matrix(
        inner_db,
        source,
        history,
        2,
        config,
        lambda: pd.DataFrame({"f": [1]}),
    )
    second = cache.full_matrix(
        outer_db,
        source,
        history,
        2,
        config,
        lambda: pd.DataFrame({"f": [2]}),
    )

    assert first["f"].tolist() == [1]
    assert second["f"].tolist() == [2]


def test_depth_cache_resets_on_new_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dfs_mod, "_build_rdb", lambda db: f"rdb::{id(db)}")
    cache = _DepthCache()
    cache.raw_rdb_for(object())
    config = CacheConfig(None, "compute")
    cache.full_matrix(
        object(),
        pd.DataFrame({"a": [1]}),
        None,
        2,
        config,
        lambda: pd.DataFrame({"f": [1]}),
    )
    cache.depth_map(None, 2, config, lambda: {"f": 2})
    assert cache._matrices and cache._depth_maps

    cache.raw_rdb_for(object())  # different db -> reset
    assert cache._matrices == {}
    assert cache._depth_maps == {}
    assert cache._rdbs == {}


# -- temporal diff (pure-pandas; no fastdfs needed) ---------------------------


def test__temporal_diff__epochtime_cols__diffed_against_cutoff_and_std_dropped() -> (
    None
):
    cutoff = pd.to_datetime(["2021-01-02", "2021-01-03"])
    event_ns = pd.to_datetime(["2021-01-01", "2021-01-01"]).astype("int64")
    df = pd.DataFrame(
        {
            "ts": cutoff,
            "users.MAX(reviews.ts_epochtime)": event_ns.astype("float64"),
            "users.STD(reviews.ts_epochtime)": [1.0, 2.0],
            "plain": [0.0, 1.0],
        }
    )
    out = _temporal_diff(df, "ts")
    assert "users.MAX(reviews.ts_epochtime)" not in out.columns
    assert "users.STD(reviews.ts_epochtime)" not in out.columns  # std dropped
    diff = out["users_MAX_reviews_ts_epochtime_diff"]
    expected = cutoff.astype("int64").astype("float64") - event_ns
    assert np.allclose(diff.to_numpy(), expected)
    assert list(out["plain"]) == [0.0, 1.0]  # untouched


def test__temporal_diff__no_cutoff_column__only_drops_std() -> None:
    df = pd.DataFrame(
        {
            "users.MAX(reviews.ts_epochtime)": [1.0],
            "users.STD(reviews.ts_epochtime)": [1.0],
        }
    )
    out = _temporal_diff(df, None)
    assert "users.MAX(reviews.ts_epochtime)" in out.columns  # kept absolute
    assert "users.STD(reviews.ts_epochtime)" not in out.columns


# --- RDB construction / featurization edge cases (real fastdfs; skip without it) ---


def _toy_task() -> SimpleNamespace:
    """An entity task over `users` (matches `_toy_db`)."""
    return SimpleNamespace(
        entity_table="users", entity_col="uid", target_col="y", time_col="ts"
    )


def _label_table(
    uids: list[int], ts: str | list[str] = "2021-06-01", y: list[float] | None = None
) -> Table:
    """A per-entity label table (the thing DFS features are built for)."""
    ts_list = [ts] * len(uids) if isinstance(ts, str) else ts
    return Table(
        df=pd.DataFrame(
            {
                "uid": uids,
                "ts": pd.to_datetime(ts_list),
                "y": y if y is not None else [float(u) for u in uids],
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="ts",
    )


def test__dfs_cache_key__tracks_inputs_not_downstream_model_choices() -> None:
    db, task = _toy_db(), _toy_task()
    anchors = _label_table([1, 2], y=[0.0, 1.0])
    history = _label_table([1, 2], y=[2.0, 3.0])
    identity = RunIdentity("rel-test", "dbfp", "task", "taskfp", phase="inner")

    key = dfs_mod._dfs_cache_key(db, task, anchors, history, 4, identity)
    changed_labels = _label_table([1, 2], y=[9.0, 8.0])
    assert dfs_mod._dfs_cache_key(db, task, changed_labels, history, 4, identity) == key
    changed_anchor = _label_table([1, 2], ts=["2021-06-01", "2021-07-01"], y=[0.0, 1.0])
    assert dfs_mod._dfs_cache_key(db, task, changed_anchor, history, 4, identity) != key
    changed_history = _label_table([1, 2], y=[7.0, 3.0])
    assert (
        dfs_mod._dfs_cache_key(db, task, anchors, changed_history, 4, identity) != key
    )
    assert dfs_mod._dfs_cache_key(db, task, anchors, history, 3, identity) != key
    assert (
        dfs_mod._dfs_cache_key(
            db, task, anchors, history, 4, identity.for_phase("outer")
        )
        != key
    )
    assert str(key).endswith("max-depth-4/matrix.parquet")


def test__dfs_depth_map_key__is_shared_across_protocol_phases() -> None:
    db, task = _toy_db(), _toy_task()
    history = _label_table([1, 2], y=[2.0, 3.0])
    identity = RunIdentity("rel-test", "dbfp", "task", "taskfp", phase="inner")

    inner = dfs_mod._dfs_depth_map_key(db, task, history, 4, identity)
    outer = dfs_mod._dfs_depth_map_key(
        db, task, history, 4, identity.for_phase("outer")
    )

    assert inner == outer


@pytest.fixture
def isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the module-level DFS cache for one test."""
    monkeypatch.setattr(dfs_mod, "_CACHE", _DepthCache())


def test__build_rdb__keyless_table__left_keyless() -> None:
    # We deliberately do NOT synthesize a PK (stay faithful to upstream); fastdfs
    # handles keyless tables internally. The cached RDB must not be mutated — that
    # invariant is enforced by build_dfs_features (see the regression test below).
    pytest.importorskip("fastdfs")
    rdb = dfs_mod._build_rdb(_toy_db())
    assert rdb.get_table_metadata("reviews").primary_key is None
    assert "__index__" not in rdb.get_table_dataframe("reviews").columns


def test__build_dfs_features__keyless_table__repeated_calls_dont_mutate_rdb(
    isolated_cache: None,
) -> None:
    # Regression for the __index__ bug: building features for two different label
    # tables (fit then predict) on a db with a keyless table must both succeed, and
    # the cached RDB must stay clean (the 2nd call raised the schema ValueError
    # pre-fix, because the depth-map step mutated the shared RDB).
    pytest.importorskip("fastdfs")
    db = _toy_db()
    task = _toy_task()
    # Hold references: the full_matrix cache keys on id(source_df), so GC'd-and-reused
    # ids of temporaries would otherwise collide (a documented _DepthCache caveat).
    train_tbl, test_tbl = _label_table([1, 2]), _label_table([3])

    f_train, _ = dfs_mod.build_dfs_features(task, db, train_tbl, depth=2, max_depth=2)
    f_test, _ = dfs_mod.build_dfs_features(task, db, test_tbl, depth=2, max_depth=2)

    assert len(f_train) == 2 and len(f_test) == 1
    cached = dfs_mod._CACHE.raw_rdb_for(db)
    assert "__index__" not in cached.get_table_dataframe("reviews").columns


def test__build_dfs_features__datetime_columns__yield_temporal_diff_features(
    isolated_cache: None,
) -> None:
    # The RDB transform pipeline (FeaturizeDatetime) is what makes timestamp
    # aggregates exist at all; the temporal-diff step then converts them to
    # time-until-cutoff. Without the pipeline, fastdfs types datetimes as
    # non-numeric and emits NO feature from them.
    pytest.importorskip("fastdfs")
    db = _toy_db()
    task = _toy_task()
    tbl = _label_table([1, 2, 3])

    feats, _ = dfs_mod.build_dfs_features(task, db, tbl, depth=2, max_depth=2)

    diff_cols = [c for c in feats.columns if c.endswith("_diff")]
    assert any("ts_epochtime" in c for c in diff_cols)  # review-time aggregates
    assert not any("_epochtime" in c and not c.endswith("_diff") for c in feats)
    assert not any("std" in c.lower() and "_epochtime" in c for c in feats.columns)

    # Value check: uid=1 has reviews at 2020-01-01 and 2020-03-01; the MAX
    # epochtime diff at cutoff 2021-06-01 is cutoff - 2020-03-01.
    max_diff_col = next(c for c in diff_cols if "MAX" in c and "ts_epochtime" in c)
    expected = float((pd.Timestamp("2021-06-01") - pd.Timestamp("2020-03-01")).value)
    row_uid1 = feats.loc[tbl.df.index[tbl.df["uid"] == 1][0]]
    assert row_uid1[max_diff_col] == pytest.approx(expected)


def test__build_dfs_features__history_table__past_label_aggregates_no_leak(
    isolated_cache: None,
) -> None:
    # With a history table, DFS derives past-target aggregates per entity. The
    # cutoff join must be strictly-less-than: an anchor whose timestamp equals a
    # history row's timestamp must NOT see that row's label (else fitting on
    # train with train-history would leak each row its own label).
    pytest.importorskip("fastdfs")
    db = _toy_db()
    task = _toy_task()
    history = _label_table(
        [1, 1, 2], ts=["2021-01-01", "2021-02-01", "2021-01-01"], y=[1.0, 3.0, 7.0]
    )
    # Anchor uid=1 at 2021-02-01: only the 2021-01-01 label (y=1.0) is history.
    anchors = _label_table([1], ts=["2021-02-01"], y=[9.0])

    feats, _ = dfs_mod.build_dfs_features(
        task, db, anchors, depth=2, max_depth=2, history_table=history
    )

    hist_cols = [c for c in feats.columns if TARGET_HISTORY_TABLE_NAME in c]
    assert hist_cols, "no target-history features were produced"
    mean_y_col = next(c for c in hist_cols if "MEAN" in c and "y" in c)
    assert feats.loc[0, mean_y_col] == pytest.approx(1.0)


def test__build_dfs_features__history_variants__cached_separately(
    isolated_cache: None,
) -> None:
    # Same split frame with and without history must not share a cached matrix.
    pytest.importorskip("fastdfs")
    db = _toy_db()
    task = _toy_task()
    history = _label_table([1, 2], ts=["2021-01-01", "2021-01-01"], y=[1.0, 2.0])
    tbl = _label_table([1, 2, 3])

    plain, _ = dfs_mod.build_dfs_features(task, db, tbl, depth=2, max_depth=2)
    with_hist, _ = dfs_mod.build_dfs_features(
        task, db, tbl, depth=2, max_depth=2, history_table=history
    )

    assert not any(TARGET_HISTORY_TABLE_NAME in c for c in plain.columns)
    assert any(TARGET_HISTORY_TABLE_NAME in c for c in with_hist.columns)
    assert dfs_mod._CACHE.matrix_computations == 2


def test__build_dfs_features__selected_anchors__match_full_matrix_rows(
    isolated_cache: None,
) -> None:
    pytest.importorskip("fastdfs")
    db, task = _toy_db(), _toy_task()
    full = _label_table(
        [1, 2, 1],
        ts=["2021-02-01", "2021-02-01", "2021-03-01"],
        y=[10.0, 20.0, 30.0],
    )
    selected_idx = np.array([2, 0])
    selected = Table(
        df=full.df.iloc[selected_idx].reset_index(drop=True),
        fkey_col_to_pkey_table=full.fkey_col_to_pkey_table,
        pkey_col=full.pkey_col,
        time_col=full.time_col,
    )

    full_features, full_cats = dfs_mod.build_dfs_features(
        task,
        db,
        full,
        depth=2,
        max_depth=2,
        history_table=full,
        keep_anchor_columns=True,
    )
    selected_features, selected_cats = dfs_mod.build_dfs_features(
        task,
        db,
        selected,
        depth=2,
        max_depth=2,
        history_table=full,
        keep_anchor_columns=True,
    )

    pd.testing.assert_frame_equal(
        full_features.iloc[selected_idx].reset_index(drop=True), selected_features
    )
    assert full_cats == selected_cats
    assert full.df.iloc[selected_idx]["y"].tolist() == selected.df["y"].tolist()
    assert any(TARGET_HISTORY_TABLE_NAME in col for col in selected_features)


def test__build_dfs_features__keep_anchor_columns__entity_and_calendar_kept(
    isolated_cache: None,
) -> None:
    pytest.importorskip("fastdfs")
    db = _toy_db()
    task = _toy_task()
    tbl = _label_table([1, 2, 3])

    dropped, dropped_cats = dfs_mod.build_dfs_features(
        task, db, tbl, depth=2, max_depth=2
    )
    assert "uid" not in dropped.columns and "ts" not in dropped.columns

    kept, cats = dfs_mod.build_dfs_features(
        task, db, tbl, depth=2, max_depth=2, keep_anchor_columns=True
    )
    assert "uid" in kept.columns and "uid" in cats  # entity key as categorical
    assert "ts" in kept.columns  # cutoff as numeric (ns) via type_columns
    for cal in ("ts.year", "ts.month", "ts.day", "ts.dayofweek"):
        assert cal in kept.columns
    assert kept.loc[0, "ts.year"] == 2021.0
    # target is never a feature
    assert "y" not in kept.columns


# -- persistent local cache layer -------------------------------------------


def _count_dfs_calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Spy on the real fastdfs materialization; returns a `{"n": int}` counter.

    `build_dfs_features` does `from fastdfs import compute_dfs_features` per call,
    so patching the attribute on the `fastdfs` module is picked up at call time.
    """
    import fastdfs

    calls = {"n": 0}
    real = fastdfs.compute_dfs_features

    def spy(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(fastdfs, "compute_dfs_features", spy)
    return calls


def test__build_dfs_features__cache_enabled__matches_uncached(
    isolated_cache: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The cached matrix must produce byte-identical features to the uncached path
    # (the parquet roundtrip preserves the matrix the downstream slice operates on).
    pytest.importorskip("fastdfs")
    db, task, tbl = _toy_db(), _toy_task(), _label_table([1, 2, 3])

    uncached, uncached_cats = dfs_mod.build_dfs_features(
        task, db, tbl, depth=2, max_depth=2
    )

    monkeypatch.setattr(dfs_mod, "_CACHE", _DepthCache())  # fresh in-process cache
    cached, cached_cats = dfs_mod.build_dfs_features(
        task,
        db,
        tbl,
        depth=2,
        max_depth=2,
        cache=CacheConfig(tmp_path, "fill"),
    )

    pd.testing.assert_frame_equal(cached, uncached)
    assert cached_cats == uncached_cats


def test__build_dfs_features__fill_mode_miss__writes_matrix(
    isolated_cache: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Under fill mode a miss computes and writes the matrix (no raise).
    pytest.importorskip("fastdfs")
    db, task, tbl = _toy_db(), _toy_task(), _label_table([1, 2, 3])
    dfs_mod.build_dfs_features(
        task,
        db,
        tbl,
        depth=2,
        max_depth=2,
        cache=CacheConfig(tmp_path, "fill"),
    )
    assert list(tmp_path.rglob("matrix.parquet"))


def test__build_dfs_features__cache_persists_across_processes(
    isolated_cache: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A second "process" (fresh in-process _CACHE) sharing the cache dir must reuse
    # the persisted matrix instead of re-running the (dominant-cost) DFS engine.
    pytest.importorskip("fastdfs")
    db, task, tbl = _toy_db(), _toy_task(), _label_table([1, 2, 3])

    calls = _count_dfs_calls(monkeypatch)
    first, _ = dfs_mod.build_dfs_features(
        task,
        db,
        tbl,
        depth=2,
        max_depth=2,
        cache=CacheConfig(tmp_path, "fill"),
    )
    assert calls["n"] == 1
    assert list(tmp_path.rglob("matrix.parquet"))

    monkeypatch.setattr(dfs_mod, "_CACHE", _DepthCache())  # simulate a new process
    second, _ = dfs_mod.build_dfs_features(
        task,
        db,
        tbl,
        depth=2,
        max_depth=2,
        cache=CacheConfig(tmp_path, "raise"),
    )
    assert calls["n"] == 1  # served from disk, DFS not re-run
    pd.testing.assert_frame_equal(second, first)


def test__build_dfs_features__cache_key_discriminates_depth_and_history(
    isolated_cache: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Distinct (max_depth, history) builds must not collide on one cache entry.
    pytest.importorskip("fastdfs")
    db, task, tbl = _toy_db(), _toy_task(), _label_table([1, 2, 3])
    history = _label_table([1, 2], ts=["2021-01-01", "2021-01-01"], y=[1.0, 2.0])

    calls = _count_dfs_calls(monkeypatch)
    # Reset the in-process cache between builds so the disk key (which the in-process
    # layer, keyed by id(source_df) not max_depth, would otherwise shadow) is what's
    # consulted — the realistic cross-process scenario.
    for kwargs in (
        {"max_depth": 2},
        {"max_depth": 4},
        {"max_depth": 2, "history_table": history},
    ):
        monkeypatch.setattr(dfs_mod, "_CACHE", _DepthCache())
        dfs_mod.build_dfs_features(
            task,
            db,
            tbl,
            depth=2,
            cache=CacheConfig(tmp_path, "fill"),
            **kwargs,
        )
    assert calls["n"] == 3  # three distinct keys, three materializations
    assert len(list(tmp_path.rglob("matrix.parquet"))) == 3


def test__build_dfs_features__cache_disabled__writes_nothing(
    isolated_cache: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("fastdfs")
    db, task, tbl = _toy_db(), _toy_task(), _label_table([1, 2, 3])

    dfs_mod.build_dfs_features(task, db, tbl, depth=2, max_depth=2)
    assert list(tmp_path.iterdir()) == []  # nothing cached outside a store


def test__build_dfs_features__leaves_no_engine_db_behind(
    isolated_cache: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # fastdfs' dfs2sql engine writes a DuckDB file under the system tempdir; if
    # build_dfs_features doesn't pin and clean an engine_path it leaks a
    # fastdfs_<uuid>.db per call. Point all tempdir usage at an empty dir and assert
    # nothing survives the call (neither our relarena_dfs_* dir nor a leaked db).
    pytest.importorskip("fastdfs")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    dfs_mod.build_dfs_features(
        _toy_task(), _toy_db(), _label_table([1, 2, 3]), depth=2, max_depth=2
    )

    assert list(tmp_path.iterdir()) == []
