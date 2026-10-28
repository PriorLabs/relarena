"""Tests for relarena.models._shared.gnn.graph.

Covers deterministic stype proposals, graphs recording the stypes they were built
with, and GNN models materializing their predict-time graph with their fit graph's
stypes and column stats. Most tests stub graph materialization out; the snapshot test
materializes two small graphs for real on CPU (no GloVe, no training).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table, TaskType

pytest.importorskip("torch_frame")

from relarena.models._shared.gnn.graph import (  # noqa: E402
    STYPE_SAMPLE_SIZE,
    build_graph,
    graph_col_to_stype,
    propose_stypes,
)
from relarena.models._shared.gnn.graph_cache import DBGraphCache  # noqa: E402


def _db(num_rows: int = 5_000) -> Database:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": np.arange(num_rows),
            # Low-cardinality integers with a long tail: depending on which rows a
            # sample hits, inference can call this categorical or numerical.
            "code": np.where(
                rng.random(num_rows) < 0.97,
                rng.integers(0, 5, num_rows),
                rng.integers(5, 400, num_rows),
            ),
            "amount": rng.normal(size=num_rows),
        }
    )
    return Database({"t": Table(df=df, fkey_col_to_pkey_table={}, pkey_col="id")})


def test__propose_stypes__same_db__samples_the_same_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each call infers from the same seeded sample, so the proposal is reproducible."""
    import torch_frame.utils

    sampled: list[tuple[int, ...]] = []

    def fake_infer(df: pd.DataFrame) -> dict[str, object]:
        sampled.append(tuple(df.index))
        return {col: torch_frame.stype.numerical for col in df.columns}

    monkeypatch.setattr(torch_frame.utils, "infer_df_stype", fake_infer)
    db = _db()

    propose_stypes(db)
    propose_stypes(db)

    assert len(sampled[0]) == STYPE_SAMPLE_SIZE
    assert sampled[0] == sampled[1]


def test__propose_stypes__same_db__same_proposal() -> None:
    db = _db()
    first = propose_stypes(db)
    assert all(propose_stypes(db) == first for _ in range(5))


def test__propose_stypes__small_table__uses_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_frame.utils

    sizes: list[int] = []
    monkeypatch.setattr(
        torch_frame.utils, "infer_df_stype", lambda df: sizes.append(len(df)) or {}
    )

    propose_stypes(_db(num_rows=20))

    assert sizes == [20]


def test__build_graph__records_the_stypes_it_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded stypes survive `make_pkey_fkey_graph` popping key columns."""
    pytest.importorskip("torch_geometric")
    from relarena.models._shared.gnn import graph as shared_graph
    from relarena.models._shared.gnn._vendor import gnn as vendor_gnn

    def fake_make_pkey_fkey_graph(
        db: Database, col_to_stype_dict: dict, **kwargs: object
    ) -> tuple[SimpleNamespace, dict]:
        col_to_stype_dict["t"].pop("id")  # what upstream does to key columns
        return SimpleNamespace(), {}

    monkeypatch.setattr(shared_graph, "make_pkey_fkey_graph", fake_make_pkey_fkey_graph)
    monkeypatch.setattr(vendor_gnn, "GloveTextEmbedding", lambda device: None)
    stypes = {"t": {"id": "numerical", "code": "categorical"}}

    data, _ = build_graph(_db(), "cpu", col_to_stype_dict=stypes)

    assert graph_col_to_stype(data) == {"t": {"id": "numerical", "code": "categorical"}}
    assert stypes == {"t": {"id": "numerical", "code": "categorical"}}  # untouched


def test__graph_col_to_stype__foreign_graph__is_none() -> None:
    assert graph_col_to_stype(SimpleNamespace()) is None


def test__graphsage_predict__materializes_with_fit_time_stypes_and_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict reuses the fit-time stypes and column stats when it rebuilds the graph.

    A predict-time database is a different object from the fit database (the RPI path
    censors it at the query anchor), so its graph is rebuilt; re-proposing stypes there
    can hand the trained column encoders a different column layout.
    """
    from relarena.models.graphsage import model as graphsage_mod
    from relarena.models.graphsage.model import GraphSAGEModel

    built_with: list[tuple[object, object]] = []

    def fake_build_graph(
        db: object,
        device: str,
        col_to_stype_dict: object = None,
        col_stats_dict: object = None,
    ) -> tuple[str, dict]:
        built_with.append((col_to_stype_dict, col_stats_dict))
        return "graph", {}

    monkeypatch.setattr(graphsage_mod, "GRAPH_CACHE", DBGraphCache())
    monkeypatch.setattr(graphsage_mod, "build_graph", fake_build_graph)
    monkeypatch.setattr(
        graphsage_mod,
        "_run_with_oom_retry",
        lambda attempt, *, start_batch_size, device, what: (
            np.zeros(3),
            start_batch_size,
        ),
    )
    model = GraphSAGEModel({})
    model._device, model._batch_size = "cpu", 8
    model._col_to_stype = {"t": {"code": "fit-time stype"}}
    model._col_stats_dict = {"t": {"code": "fit-time stats"}}

    model.predict(task=None, db=object(), table=None)

    assert built_with == [(model._col_to_stype, model._col_stats_dict)]


def test__graphsage__fit_early_predict_later__keeps_fit_category_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later snapshot is encoded with the category indices the model was fit on.

    torch_frame indexes categories by frequency. After the fit cutoff EF overtakes AB
    and YZ, and a new category QQ becomes the most frequent, so the later snapshot's
    own stats would give the trained encoders AB=2, YZ=3, EF=1, QQ=0.
    """
    pytest.importorskip("torch_geometric")
    from torch_frame import stype

    from relarena.models._shared.gnn._vendor import gnn as vendor_gnn
    from relarena.models.graphsage import model as graphsage_mod

    fit_codes = ["AB"] * 50 + ["YZ"] * 30 + ["EF"] * 20
    codes = fit_codes + ["EF"] * 70 + ["QQ"] * 120
    df = pd.DataFrame(
        {
            "id": np.arange(len(codes)),
            "ts": pd.date_range("2020-01-01", periods=len(codes), freq="D"),
            "code": codes,
        }
    )
    db = Database(
        {"t": Table(df=df, fkey_col_to_pkey_table={}, pkey_col="id", time_col="ts")}
    )
    graph_cache = DBGraphCache()
    monkeypatch.setattr(graphsage_mod, "GRAPH_CACHE", graph_cache)
    monkeypatch.setattr(vendor_gnn, "GloveTextEmbedding", lambda device: None)
    # Skip training and inference: only the graphs fit and predict build matter here.
    monkeypatch.setattr(
        graphsage_mod,
        "_run_with_oom_retry",
        lambda attempt, *, start_batch_size, device, what: (None, start_batch_size),
    )
    model = graphsage_mod.GraphSAGEModel(dict(graphsage_mod._DEFAULT_CONFIG))
    task = SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION)

    model.fit(task, db.upto(df["ts"][len(fit_codes) - 1]), None, None, seed=0)
    model.predict(task, db, None)

    data, _ = graph_cache.get(db, lambda: pytest.fail("predict built no graph"))
    tf = data["t"].tf
    code_col = tf.col_names_dict[stype.categorical].index("code")
    fit_index = {"AB": 0, "YZ": 1, "EF": 2}  # QQ is unseen at fit: missing (-1)
    assert tf.feat_dict[stype.categorical][:, code_col].tolist() == [
        fit_index.get(code, -1) for code in codes
    ]
