"""Tests for relarena.models._shared.gnn.graph.

Covers deterministic stype proposals, graphs recording the stypes they were built
with, and GNN models materializing their predict-time graph with their fit graph's
stypes. Graph materialization itself needs the `rdl` extra and is exercised by the GPU
smoke run; these tests stub it out.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

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
    import relbench.modeling.graph as relbench_graph

    from relarena.models._shared.gnn._vendor import gnn as vendor_gnn

    def fake_make_pkey_fkey_graph(
        db: Database, col_to_stype_dict: dict, text_embedder_cfg: object
    ) -> tuple[SimpleNamespace, dict]:
        col_to_stype_dict["t"].pop("id")  # what upstream does to key columns
        return SimpleNamespace(), {}

    monkeypatch.setattr(
        relbench_graph, "make_pkey_fkey_graph", fake_make_pkey_fkey_graph
    )
    monkeypatch.setattr(vendor_gnn, "GloveTextEmbedding", lambda device: None)
    stypes = {"t": {"id": "numerical", "code": "categorical"}}

    data, _ = build_graph(_db(), "cpu", col_to_stype_dict=stypes)

    assert graph_col_to_stype(data) == {"t": {"id": "numerical", "code": "categorical"}}
    assert stypes == {"t": {"id": "numerical", "code": "categorical"}}  # untouched


def test__graph_col_to_stype__foreign_graph__is_none() -> None:
    assert graph_col_to_stype(SimpleNamespace()) is None


def test__graphsage_predict__materializes_with_fit_time_stypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict reuses the fit-time stypes when it rebuilds the graph.

    A predict-time database is a different object from the fit database (the RPI path
    censors it at the query anchor), so its graph is rebuilt; re-proposing stypes there
    can hand the trained column encoders a different column layout.
    """
    from relarena.models._shared.gnn import graph as shared_graph
    from relarena.models.graphsage import model as graphsage_mod
    from relarena.models.graphsage.model import GraphSAGEModel

    built_with: list[object] = []

    def fake_build_graph(
        db: object, device: str, col_to_stype_dict: object = None
    ) -> tuple[str, dict]:
        built_with.append(col_to_stype_dict)
        return "graph", {}

    monkeypatch.setattr(graphsage_mod, "GRAPH_CACHE", DBGraphCache())
    monkeypatch.setattr(shared_graph, "build_graph", fake_build_graph)
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

    model.predict(task=None, db=object(), table=None)

    assert built_with == [model._col_to_stype]
