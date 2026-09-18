"""Real DFS, temporal tuning and prediction with an in-memory estimator backend."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from relarena_core.featurization import dfs
from relarena_core.tfm import TFMSpec

from tabpfn_rel import PredictiveContext, PredictiveQuery, PredictiveQuerySpec, tfm


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


_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tiny_database.py"


@pytest.fixture(scope="session")
def write_database() -> Callable[..., Path]:
    """The generated-database writer from the example script, loaded by path."""
    spec = importlib.util.spec_from_file_location("tiny_database", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_database


@pytest.fixture
def query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    write_database: Callable[..., Path],
) -> PredictiveContext:
    for variable in (
        "RELARENA_CACHE_DIR",
        "RELARENA_DISABLE_CACHE",
        "RELARENA_DISABLE_FEATURE_CACHE",
    ):
        monkeypatch.delenv(variable, raising=False)
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
    return PredictiveContext(spec, data_version="test-v1")


@pytest.mark.parametrize(
    "query", ["binary_classification", "regression"], indirect=True
)
@pytest.mark.parametrize("backend", ["local", "client"])
@pytest.mark.parametrize("n_trials", [0, 2])
def test_rpi_fits_tunes_and_reuses_prediction_cache(
    query: PredictiveContext, backend: str, n_trials: int, tmp_path: Path
) -> None:
    fitted = query.fit(
        f"tabpfn-rel-{backend}", n_trials=n_trials, cache_dir=tmp_path / "cache"
    )
    predictions = fitted.predict(
        PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    )
    pd.testing.assert_frame_equal(
        predictions,
        fitted.predict(PredictiveQuery(entities="all", at_timestamp="test_timestamp")),
    )
    assert sorted(predictions["customer_id"]) == ["a", "b", "c", "d"]
    assert predictions["y_pred"].notna().all()
    assert len(query.compute_test_labels()) == 4
    assert fitted.config["max_depth"] in (2, 3)
    if n_trials:
        assert len(fitted.trials) == 2
        assert all(trial.val_score is not None for trial in fitted.trials)
    else:
        assert fitted.trials is None
    if backend == "client":
        assert "description__raw_text" in fitted._model._fitted.estimator.columns_
    assert list((tmp_path / "cache").rglob("*.parquet"))


def test_cli_benchmarks_discovered_model(
    query: PredictiveContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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


def test_context_cache_reused_across_queries_and_fits(
    query: PredictiveContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "shared-cache"
    query.precompute_cache(cache_dir)
    training_files = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in cache_dir.rglob("*.parquet")
    }
    assert training_files
    build_rdb = dfs._build_rdb

    def forbid_build(*args: object, **kwargs: object) -> None:
        pytest.fail("Cache hit rebuilt DFS features")

    monkeypatch.setattr(dfs, "_CACHE", dfs._DepthCache())
    monkeypatch.setattr(dfs, "_build_rdb", forbid_build)
    fitted = query.fit("tabpfn-rel-local", n_trials=2, cache_dir=cache_dir)
    first_model = fitted._model
    monkeypatch.setattr(dfs, "_build_rdb", build_rdb)
    first = PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    second = PredictiveQuery(entities=["b"], at_timestamp="2004-11-15")
    first_predictions = fitted.predict(first)
    second_predictions = fitted.predict(second)
    assert second_predictions.customer_id.tolist() == ["b"]
    assert second_predictions.date.tolist() == [pd.Timestamp("2004-11-15")]
    assert fitted._model is first_model
    assert all(
        (path.stat().st_mtime_ns, path.read_bytes()) == original
        for path, original in training_files.items()
    )
    all_files = {path: path.stat().st_mtime_ns for path in cache_dir.rglob("*.parquet")}
    monkeypatch.setattr(dfs, "_CACHE", dfs._DepthCache())
    monkeypatch.setattr(dfs, "_build_rdb", forbid_build)
    pd.testing.assert_frame_equal(first_predictions, fitted.predict(first))
    pd.testing.assert_frame_equal(second_predictions, fitted.predict(second))
    explicit = PredictiveQuery(entities="all", at_timestamp="2004-12-01")
    pd.testing.assert_frame_equal(first_predictions, fitted.predict(explicit))
    fresh = PredictiveContext.from_yaml(
        tmp_path / "task.yaml", data_dir=tmp_path, data_version="test-v1"
    )
    other = fresh.fit("tabpfn-rel-client", n_trials=0, cache_dir=cache_dir)
    assert other._model is not first_model
    pd.testing.assert_frame_equal(first_predictions, fitted.predict(first))
    assert {
        path: path.stat().st_mtime_ns for path in cache_dir.rglob("*.parquet")
    } == all_files
