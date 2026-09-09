"""Real DFS, temporal tuning and prediction with an in-memory estimator backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from relarena_core.tfm import TFMSpec

from examples.tiny_database import write_database
from tabpfn_rel import PredictiveQuery, PredictiveQuerySpec, tfm


class _Estimator:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _Estimator:
        assert len(X) == len(y) > 0
        assert X.shape[1] > 0
        self.classes_ = np.unique(y)
        self.mean_ = float(np.mean(y))
        self.columns_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


def _make_estimator(**kwargs: object) -> _Estimator:
    return _Estimator()


@pytest.fixture
def query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> PredictiveQuery:
    for name in ("tabpfn-v3", "tabpfn-v3-api"):
        monkeypatch.setitem(
            tfm.TFM_REGISTRY,
            name,
            TFMSpec(_make_estimator, _make_estimator, 100_000, supports_text=True),
        )
    task_path = write_database(
        tmp_path, getattr(request, "param", "binary_classification")
    )
    spec = PredictiveQuerySpec.from_yaml(str(task_path), data_dir=str(tmp_path))
    return PredictiveQuery(spec, data_version="test-v1")


@pytest.mark.parametrize(
    "query", ["binary_classification", "regression"], indirect=True
)
@pytest.mark.parametrize("backend", ["local", "client"])
@pytest.mark.parametrize("n_trials", [0, 2])
def test_rpi_fits_tunes_and_reuses_prediction_cache(
    query: PredictiveQuery, backend: str, n_trials: int, tmp_path: Path
) -> None:
    query.fit(f"tabpfn-rel-{backend}", n_trials=n_trials, cache_dir=tmp_path / "cache")
    predictions = query.predict()
    pd.testing.assert_frame_equal(predictions, query.predict())
    assert sorted(predictions["customer_id"]) == ["a", "b", "c", "d"]
    assert predictions["y_pred"].notna().all()
    assert len(query.compute_test_labels()) == 4
    assert query.config["max_depth"] in (2, 3)
    if n_trials:
        assert len(query.trials) == 2
        assert all(trial.val_score is not None for trial in query.trials)
    else:
        assert query.trials is None
    if backend == "client":
        assert "description__raw_text" in query._model._fitted.estimator.columns_
    assert list((tmp_path / "cache").rglob("*.parquet"))


def test_cli_benchmarks_discovered_model(
    query: PredictiveQuery, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = pytest.importorskip("relarena.cli")
    runner = pytest.importorskip("relarena.runner")
    from relarena.tasks import TaskSpec

    source = query._source
    monkeypatch.setattr(runner, "RelBenchDatasetTask", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        cli,
        "list_entity_tasks",
        lambda datasets: [TaskSpec("tiny", "customers", query.task.task_type)],
    )
    output = tmp_path / "results.csv"
    assert (
        cli.main(
            [
                "--model",
                "tabpfn-rel-local",
                "--datasets",
                "tiny",
                "--n-trials",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    results = pd.read_csv(output)
    assert results["test_score"].notna().any()
