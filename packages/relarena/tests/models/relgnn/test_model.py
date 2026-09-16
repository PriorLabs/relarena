"""Unit tests for the RelGNN model.

Covers the vendored model core (the dependency-free `get_atomic_routes` route
builder), the adapter's registration / search space / default config, and the disk
graph-cache key — all without the `graphsage` extra, since the heavy graph stack is
imported lazily inside `fit` / `predict`. The full fit/predict graph path runs
end-to-end against a real RelBench dataset on a GPU (the CLI smoke run), not here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from ConfigSpace import Configuration
from relbench.base import TaskType

import relarena.models.relgnn.preprocessing as preprocessing
from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.models._shared.gnn.graph_cache import DBGraphCache
from relarena.models.relgnn._vendor.atomic_routes import get_atomic_routes
from relarena.models.relgnn.model import (
    _DEFAULT_CONFIG,
    RELGNN_SPACE,
    RelGNNEarlyStopModel,
    RelGNNModel,
)
from relarena.models.relgnn.preprocessing import (
    _GRAPH_CACHE_VERSION,
    _validate_graph,
    graph_cache_key,
    load_graph,
)
from relarena.registry import registry

# -- atomic routes (pure topology; no heavy deps) ---------------------------


def test__get_atomic_routes__single_fkey__emits_dim_dim_and_reverse() -> None:
    # A fact table with exactly one foreign key yields a direct dim-dim route plus
    # its reverse; non-`f2p` edges (e.g. existing reverse edges) are ignored.
    edge_types = [
        ("order", "f2p_customer", "customer"),
        ("customer", "rev_f2p_customer", "order"),  # ignored: not an f2p edge
    ]
    assert set(get_atomic_routes(edge_types)) == {
        ("dim-dim", "order", "f2p_customer", "customer"),
        ("dim-dim", "customer", "rev_f2p_customer", "order"),
    }


def test__get_atomic_routes__shared_fact__emits_dim_fact_dim_pairs() -> None:
    # Two foreign keys on one fact table compose into a direct dim->dim route in each
    # direction: attend along one fkey, SAGE-aggregate the fact via the other's reverse.
    edge_types = [
        ("review", "f2p_customer", "customer"),
        ("review", "f2p_product", "product"),
    ]
    assert set(get_atomic_routes(edge_types)) == {
        (
            "dim-fact-dim",
            "review",
            "f2p_customer",
            "customer",
            "product",
            "rev_f2p_product",
            "review",
        ),
        (
            "dim-fact-dim",
            "review",
            "f2p_product",
            "product",
            "customer",
            "rev_f2p_customer",
            "review",
        ),
    }


def test__get_atomic_routes__self_loop_fkey__disambiguates_then_emits_dim_dim() -> None:
    # A self-referential fkey (src == dst) must not collapse with the table's other
    # fkeys: it is keyed separately, so it still produces a lone dim-dim route plus
    # its reverse.
    edge_types = [
        ("post", "f2p_parent", "post"),
        ("post", "f2p_author", "user"),
    ]
    routes = set(get_atomic_routes(edge_types))
    assert ("dim-dim", "post", "f2p_parent", "post") in routes
    assert ("dim-dim", "post", "rev_f2p_parent", "post") in routes
    # The non-self fkey is alone under its key too, so it is dim-dim, not dim-fact-dim.
    assert ("dim-dim", "post", "f2p_author", "user") in routes
    assert not any(r[0] == "dim-fact-dim" for r in routes)


# -- vendored model import (needs the graphsage extra: PyG / PyTorch Frame) --


def test__relgnn_model__importable() -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_frame")
    from relarena.models.relgnn._vendor.model import RelGNN_Model  # noqa: F401


# -- registration & search space --------------------------------------------


def test__relgnn__registered_under_name() -> None:
    assert "relgnn" in registry
    assert registry.get("relgnn") is RelGNNModel
    assert registry.search_space("relgnn") is RELGNN_SPACE


def test__relgnn_es__registered_early_stop_variant() -> None:
    # Same model + search space as relgnn, only the final-fit regime differs.
    assert registry.get("relgnn-es") is RelGNNEarlyStopModel
    assert registry.search_space("relgnn-es") is RELGNN_SPACE
    assert RelGNNEarlyStopModel.refit_on_full_data is False
    assert RelGNNModel.refit_on_full_data is True


def test__relgnn__supports_binary_and_regression_only() -> None:
    assert RelGNNModel.supported_task_types == frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )


def test__default_config__matches_modal_per_task_values() -> None:
    # The modal per-task config across the 21 RelBench entity tasks.
    assert _DEFAULT_CONFIG == {
        "channels": 128,
        "num_model_layers": 2,
        "num_heads": 8,
        "aggr": "sum",
        "num_neighbors": 128,
        "subgraph_type": "directional",
        "simplified_MP": False,
        "lr": 0.005,
    }


def test__search_space__samples_carry_all_tuned_keys() -> None:
    # Every sampled config must define the keys fit reads directly (all but the
    # default-only simplified_MP).
    cfg = RELGNN_SPACE.space.sample_configuration()
    assert isinstance(cfg, Configuration)
    assert set(cfg.keys()) == {
        "channels",
        "num_model_layers",
        "num_heads",
        "aggr",
        "num_neighbors",
        "subgraph_type",
        "lr",
    }


def test__search_space__default_first_then_samples() -> None:
    configs = RELGNN_SPACE.configs(n_trials=3, seed=0)
    assert configs[0] == _DEFAULT_CONFIG
    assert len(configs) == 4  # default + 3 samples


# -- graph cache key ---------------------------------------------------------


def _fake_db(*, cols: tuple[str, ...], max_day: int) -> SimpleNamespace:
    # Identity-backed key tests do not inspect rows; this only supplies a db object.
    table = SimpleNamespace(
        df=SimpleNamespace(columns=list(cols)),
        pkey_col="id",
        time_col="time",
        fkey_col_to_pkey_table={"u": "users"},
    )
    ts = pd.Timestamp(f"2020-01-{max_day:02d}")
    return SimpleNamespace(table_dict={"events": table}, max_timestamp=ts)


def test__graph_cache_key__deterministic_for_same_db() -> None:
    db = _fake_db(cols=("id", "time", "x"), max_day=5)
    identity = RunIdentity("dataset", "dbfp", "task-a", "taskfp", phase="inner")
    assert graph_cache_key(db, identity) == graph_cache_key(db, identity)


def test__graph_cache_key__differs_by_phase() -> None:
    db = _fake_db(cols=("id", "time", "x"), max_day=5)
    identity = RunIdentity("dataset", "dbfp", "task", "taskfp", phase="inner")
    assert graph_cache_key(db, identity) != graph_cache_key(
        db, identity.for_phase("outer")
    )


def test__graph_cache_key__predict_phase__differs_by_censored_view() -> None:
    identity = RunIdentity("user", "schema", "task", "taskfp", phase="predict")
    first = _fake_db(cols=("id", "time", "x"), max_day=5)
    second = _fake_db(cols=("id", "time", "x"), max_day=6)

    assert graph_cache_key(first, identity) != graph_cache_key(second, identity)


def test__graph_cache_key__omits_task_identity_for_cross_task_sharing() -> None:
    db = _fake_db(cols=("id", "time", "x"), max_day=5)
    first = RunIdentity("dataset", "dbfp", "task-a", "one", phase="inner")
    second = RunIdentity("dataset", "dbfp", "task-b", "two", phase="inner")
    assert graph_cache_key(db, first) == graph_cache_key(db, second)


def test__graph_cache_key__carries_version_prefix() -> None:
    db = _fake_db(cols=("id", "time", "x"), max_day=5)
    identity = RunIdentity("dataset", "dbfp", None, None, phase="inner")
    assert graph_cache_key(db, identity).startswith(
        f"relgnn/graph-v{_GRAPH_CACHE_VERSION}/"
    )


def test__load_graph__fill_then_hit__publishes_complete_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = object()
    identity = RunIdentity("dataset", "dbfp", "ignored", "ignored", phase="inner")
    calls: list[Path] = []

    def materialize(db: object, path: Path) -> tuple[dict, dict]:
        calls.append(path)
        graph = path / "graph.pt"
        if not graph.exists():
            graph.write_text("graph")
        return ({"graph": graph.read_text()}, {})

    monkeypatch.setattr(preprocessing, "_materialize", materialize)
    monkeypatch.setattr(preprocessing, "_validate_graph", lambda *args: None)
    monkeypatch.setattr(preprocessing, "_GRAPH_CACHE", DBGraphCache())
    filled = load_graph(
        db,
        CacheConfig(tmp_path, "fill"),
        identity,  # type: ignore[arg-type]
    )
    artifact = tmp_path / graph_cache_key(db, identity)  # type: ignore[arg-type]
    assert (artifact / "_COMPLETE").is_file()

    monkeypatch.setattr(preprocessing, "_GRAPH_CACHE", DBGraphCache())
    hit = load_graph(
        db,
        CacheConfig(tmp_path, "raise"),
        identity,  # type: ignore[arg-type]
    )
    assert hit == filled
    assert calls == [calls[0], artifact]


def test__load_graph__incomplete_directory__fails_without_materializing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = object()
    identity = RunIdentity("dataset", "dbfp", None, None, phase="inner")
    artifact = tmp_path / graph_cache_key(db, identity)  # type: ignore[arg-type]
    artifact.mkdir(parents=True)
    monkeypatch.setattr(preprocessing, "_GRAPH_CACHE", DBGraphCache())
    monkeypatch.setattr(
        preprocessing,
        "_materialize",
        lambda *args: pytest.fail("incomplete hit must not materialize"),
    )
    with pytest.raises(ValueError, match="incomplete RelGNN graph"):
        load_graph(
            db,
            CacheConfig(tmp_path, "raise"),
            identity,  # type: ignore[arg-type]
        )


# -- graph integrity guard (catches silently-corrupt builds before caching) --


def _fake_graph(year_rows: list[int], min_year: int = 2015) -> tuple[object, dict]:
    # Minimal stand-in for a relbench HeteroData + col_stats: one node type with one
    # timestamp column, year at component 0 of the timestamp feature.
    import torch
    import torch_frame
    from torch_frame.data.stats import StatType

    feat = torch.tensor(year_rows).reshape(-1, 1, 1)  # (#rows, #ts cols=1, comps>=1)
    tf = SimpleNamespace(
        col_names_dict={torch_frame.stype.timestamp: ["ts"]},
        feat_dict={torch_frame.stype.timestamp: feat},
    )

    class _Data:
        node_types = ["ev"]

        def __getitem__(self, _key: str) -> SimpleNamespace:
            return SimpleNamespace(tf=tf)

    col_stats = {"ev": {"ts": {StatType.YEAR_RANGE: (min_year, max(year_rows))}}}
    return _Data(), col_stats


def test__validate_graph__clean_years__passes() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_frame")
    _validate_graph(*_fake_graph([2015, 2016, 2017], min_year=2015))  # no raise


def test__validate_graph__year_below_stats_min__raises() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_frame")
    # A zeroed row (year 0) below the column's min (2015) — the avito corruption mode.
    with pytest.raises(ValueError, match="corrupt graph"):
        _validate_graph(*_fake_graph([2015, 0, 2015], min_year=2015))


def test__load_graph__memoizes_by_db_identity__one_build_per_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_load_graph` reuses one in-memory graph per db so a sweep doesn't reload it.

    Repeated calls for one database build once (the OOM fix); a new database identity
    rebuilds, since the cache holds one graph at a time (the inner -> outer split).
    """
    monkeypatch.setattr(preprocessing, "_GRAPH_CACHE", DBGraphCache())
    built: list[object] = []

    def fake_cached(*args: object, **kwargs: object) -> tuple[dict, dict]:
        built.append(args[0])
        return ({"n": len(built)}, {})

    monkeypatch.setattr(preprocessing, "cached_artifact", fake_cached)

    db = object()
    identity = RunIdentity("dataset", "db", None, None, phase="inner")
    config = CacheConfig(None, "compute")
    first = load_graph(db, config, identity)  # type: ignore[arg-type]
    second = load_graph(db, config, identity)  # type: ignore[arg-type]
    assert first is second
    assert len(built) == 1  # reused across trials, not reloaded

    load_graph(object(), config, identity)  # type: ignore[arg-type]
    assert len(built) == 2
