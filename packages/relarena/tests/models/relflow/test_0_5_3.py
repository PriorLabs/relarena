"""Check relflow 0.5.3 presets, outcome availability, and full refits."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table, TaskType

from relarena.models.relflow import RelFlow053Model, RelFlowModel, search_space_053
from relarena_core.dataset import OuterSplit, concat_tables
from relarena_core.identity import RunIdentity
from relarena_core.registry import registry
from relarena_core.search_space import TaskStats

Fixture = tuple[SimpleNamespace, Database, Table, Table, Table]


@pytest.mark.parametrize(
    "rows,batch,passes,cross",
    [
        (9_999, 32, 7, False),
        (10_000, 32, 8, True),
        (19_999, 32, 8, True),
        (20_000, 256, 2, False),
        (500_000, 256, 2, False),
        (3_386_276, 256, 2, False),
    ],
)
def test__preset__inner_row_count__one_fixed_config(
    rows: int, batch: int, passes: int, cross: bool
) -> None:
    assert registry.get("relflow-2026-11-04") is RelFlowModel
    assert registry.get("relflow-latest") is RelFlow053Model
    space = search_space_053(TaskStats(rows))
    config = space.default_overrides
    assert space.configs(n_trials=50, seed=17) == [config]
    assert config["training_rows"] == rows
    assert config["execution"]["batch_size"] == config["recipe"]["batch_size"] == batch
    assert config["execution"]["workers"] == 0
    assert config["recipe"]["passes"] == passes
    assert config["recipe"]["root_cross"] is config["recipe"]["branch_cross"] is cross
    assert config["text_token_cap"] == 128
    if rows >= 20_000:
        expected = search_space_053(TaskStats(20_000)).default_overrides
        assert config == {**expected, "training_rows": rows}


def anchors(ids: list[int], days: list[int], labels: list[int] | None = None) -> Table:
    frame = pd.DataFrame(
        {
            "entity": ids,
            "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D"),
        }
    )
    if labels is not None:
        frame["value"] = labels
    return Table(frame, {"entity": "people"}, time_col="date")


@pytest.fixture
def temporal_task() -> Fixture:
    task = SimpleNamespace(
        entity_table="people",
        entity_col="entity",
        target_col="value",
        time_col="date",
        task_type=TaskType.REGRESSION,
        timedelta=pd.Timedelta(days=2),
        metrics=[],
        dataset=SimpleNamespace(
            val_timestamp=pd.Timestamp("2020-01-10"),
            test_timestamp=pd.Timestamp("2020-01-15"),
        ),
    )
    db = Database(
        {
            "people": Table(
                pd.DataFrame(
                    {"id": [0, 1, 2], "team": [0, 0, 1], "amount": [0.2, 0.4, 0.6]}
                ),
                {"team": "teams"},
                pkey_col="id",
            ),
            "teams": Table(pd.DataFrame({"team_id": [0, 1]}), {}, pkey_col="team_id"),
            "events": Table(
                pd.DataFrame(
                    {
                        "id": [0, 1, 2],
                        "owner": [0, 0, 1],
                        "t": pd.to_datetime(["2020-01-02", "2020-01-05", "2020-01-01"]),
                        "amount": [2.1, 3.0, 7.2],
                    }
                ),
                {"owner": "people"},
                pkey_col="id",
                time_col="t",
            ),
        }
    )
    train = anchors([0, 0, 0, 1, 2], [0, 2, 4, 0, 1], [0, 2, 4, 20, 10])
    val = anchors([0, 1], [9, 9], [8, 40])
    query = anchors([1, 0, 2], [15, 15, 16])
    return task, db, train, val, query


def test__outcomes__anchor_horizon_and_split__exact_mean_without_self_leakage(
    temporal_task: Fixture, tmp_path: Path
) -> None:
    pytest.importorskip("relflow", minversion="0.5.3")
    import pyarrow.parquet as pq

    from relarena.models.relflow.context import Context, infer, materialize
    from relarena.models.relflow.history import History

    task, db, train, val, _ = temporal_task
    config = search_space_053(TaskStats(len(train))).default_overrides
    ctx = Context.open(db, task.dataset.val_timestamp, tmp_path, threads=1)
    try:
        structure = replace(infer(ctx, task, config), target_kind="ordinal")
        history = History.fit(ctx, task, concat_tables(train, val), structure, "outer")
        ctx.freeze()
        history.prepare(ctx)
        table = anchors([0, 0, 0, 0, 2], [1, 2, 4, 16, 0], [-9999] * 5)

        def rows(labelled: bool) -> list[dict[str, Any]]:
            path = materialize(
                ctx,
                structure,
                task,
                table,
                tmp_path / f"rows-{labelled}.parquet",
                labelled=labelled,
                history=history,
            )
            return pq.read_table(path).to_pylist()

        pred = rows(False)
        assert [r[history.outcome_mean] for r in pred] == [None, 0, 1, 3.5, None]
        assert [v["value"] for v in pred[3][history.history]] == [0, 2, 4, 8]
        peers = next(c for c in history.cohorts if c.table == "people")
        assert pred[2][peers.mean] == 20 and pred[3][peers.mean] == 40
        assert all("target" not in r for r in pred)
        assert rows(True)[3][history.outcome_mean] == 2
    finally:
        ctx.close()


@pytest.mark.parametrize("row_count", [9_999, 10_000, 20_000])
def test__registered_full_refit__same_seed__replays_on_cpu(
    temporal_task: Fixture, monkeypatch: pytest.MonkeyPatch, row_count: int
) -> None:
    pytest.importorskip("relflow", minversion="0.5.3")
    import torch

    from relarena.refit import refit_and_evaluate

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    task, db, train, val, query = temporal_task

    def evaluate(
        pred: np.ndarray, table: Table | None, metrics: list[Callable]
    ) -> dict[str, float]:
        assert table is None and "value" not in query.df
        return {
            metric.__name__: float(metric(np.array([30, 8, 10]), pred))
            for metric in metrics
        }

    task.evaluate = evaluate
    # Exercise each unchanged recipe on a tiny split; row_count picks the preset.
    config = search_space_053(TaskStats(row_count)).default_overrides
    split = OuterSplit(
        db_state=db,
        cutoff=task.dataset.test_timestamp,
        train_table=train,
        eval_table=query,
        val_table=val,
    )
    identity = RunIdentity("people", None, "future-value", None, phase="outer")

    def run() -> np.ndarray:
        return refit_and_evaluate(
            RelFlow053Model, config, task, split, seed=0, run_identity=identity
        )["test_pred"]

    first, second = run(), run()
    assert first.shape == (3,) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("kind,minimum", [("ordinal", 1.0), ("number", 5e-324)])
def test__hurdle__five_percent_zeros_and_weighted_tie__round_trip(
    kind: str, minimum: float
) -> None:
    pytest.importorskip("relflow", minversion="0.5.3")
    import polars as pl

    from relarena.models.relflow.hurdle import (
        Hurdle,
        positive_estimate,
        positive_target,
    )

    assert Hurdle.fit(np.array([0] + [1] * 19), target_kind=kind) == Hurdle(kind)
    for ineligible in ([0] + [1] * 20, [0, 0], [-1, 0, 1]):
        assert Hurdle.fit(np.array(ineligible), target_kind=kind) is None
    fit_rows = positive_target(pl.DataFrame({"target": [0, 3]}))
    assert fit_rows["target"].to_list() == [None, 3]
    frame = pl.DataFrame(
        {
            "inputs": [{"id": i} for i in range(3)],
            "predictions": [
                {"target": {"state": {"null": n, "valued": v}, "content": c}}
                for n, v, c in [(0.25, 0.5, 3.0), (0.125, 0.5, 2.0), (0.0, 1.0, -2.0)]
            ],
        }
    )
    result = positive_estimate.partial(odds_scale=2.0, minimum=minimum)(frame)
    assert result["estimate"].to_list() == [0.0, 2.0, minimum]
    np.testing.assert_allclose(result["p_zero"], [0.5, 1 / 3, 0.0])


def test__text__complete_cached_revision__resolves_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("relflow", minversion="0.5.3")
    import huggingface_hub.constants as constants

    from relarena.models.relflow.training import (
        ENCODER,
        ENCODER_FILES,
        ENCODER_REVISION,
        encoder_snapshot,
    )

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path
        / ("models--" + ENCODER.replace("/", "--"))
        / "snapshots"
        / ENCODER_REVISION
    )
    snapshot.mkdir(parents=True)
    for filename in ENCODER_FILES:
        (snapshot / filename).touch()
    assert Path(encoder_snapshot(ENCODER, individual_files=True)) == snapshot
