"""Tests for the copy-ready RelArena model adapter."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table, TaskType

from relarena.models.nori_rel import model as model_module
from relarena.models.nori_rel.model import NoriRel
from relarena_core.registry import registry
from relarena_core.search_space import TaskStats, resolve_search_space


class FakeNoriRegressor:
    """Small estimator double for contract tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.fit_rows = 0
        self.output_type = ""
        self.memory_report_ = {
            **kwargs["memory_policy"],
            "gpu_budget_absolute_gb": None,
            "host_budget_absolute_gb": None,
            "offload_to_host": False,
            "rung": "no_cache",
            "dropped_context_rows": 0,
        }

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> FakeNoriRegressor:
        self.fit_rows = len(features)
        self.fit_features = features.copy()
        self.target = target
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        output_type: str,
    ) -> np.ndarray:
        self.predict_features = features.copy()
        self.output_type = output_type
        return np.resize(np.array([-0.25, 0.5, 1.25]), len(features))


def _install_fake_nori(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_module,
        "_load_nori",
        lambda: (FakeNoriRegressor, nullcontext),
    )
    monkeypatch.setattr(
        model_module,
        "_checkpoint_path",
        lambda: "/nori-30m.pt",
    )


def test_registration_and_fixed_space() -> None:
    assert registry.get("nori-rel") is NoriRel
    assert NoriRel.supported_task_types == {
        TaskType.REGRESSION,
        TaskType.BINARY_CLASSIFICATION,
    }
    space = resolve_search_space(
        registry.search_space_for(NoriRel),
        TaskStats(num_train_nodes=3),
    )
    assert space.configs(10, seed=0) == model_module.ROUTED_GRID
    assert [
        {k: v for k, v in arm.items() if k != "classification_backend"}
        for arm in model_module.ROUTED_GRID
    ] == model_module.NORIDFS_GRID
    assert all(
        arm["classification_backend"] == model_module.FASTDFS_BACKEND
        for arm in model_module.ROUTED_GRID
    )


@pytest.mark.parametrize("arm", model_module.ROUTED_GRID)
@pytest.mark.parametrize(
    ("task_type", "routes_to_shared_dfs"),
    [(TaskType.REGRESSION, False), (TaskType.BINARY_CLASSIFICATION, True)],
)
def test_entry_routes_the_feature_backend_by_task_type(
    monkeypatch: pytest.MonkeyPatch,
    arm: dict[str, Any],
    task_type: TaskType,
    routes_to_shared_dfs: bool,
) -> None:
    """Regression keeps its NoriDFS arm; classification always gets shared DFS."""
    seen: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        model_module.NoriRelModel,
        "fit",
        lambda self, *args, **kwargs: seen.update(config=dict(self.config)),
    )

    NoriRel(dict(arm)).fit(
        SimpleNamespace(task_type=task_type), object(), object(), None, seed=0
    )

    expected = model_module.DEFAULT_CONFIG if routes_to_shared_dfs else arm
    assert seen["config"] == expected


def test_all_missing_training_columns_leave_fit_and_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column empty in the training context reaches neither fit nor predict."""

    def fake_features(
        task: Any, db: Any, table: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db, kwargs
        rows = len(table.df)
        # Empty while fitting, populated with strings at predict.
        sparse = [None] * rows if rows == 3 else ["seen"] * rows
        frame = pd.DataFrame({"value": np.arange(rows, dtype=float), "sparse": sparse})
        return frame, ["sparse"]

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", fake_features)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    task = SimpleNamespace(
        target_col="target",
        time_col=None,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(2)))
    model = NoriRel(dict(model_module.DEFAULT_CONFIG))

    model.fit(task, object(), train, None, seed=0)
    model.predict(task, object(), query)

    assert model._columns == ["value"]
    assert model._model.kwargs["categorical_columns"] == []
    assert list(model._model.fit_features) == ["value"]
    assert list(model._model.predict_features) == ["value"]


def test_text_column_empty_in_training_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A text column made only of stringified nulls is empty, not prose."""

    def fake_features(
        task: Any, db: Any, table: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db, kwargs
        return pd.DataFrame({"value": np.arange(len(table.df), dtype=float)}), []

    def fake_attach(
        features: pd.DataFrame,
        db: Any,
        task: Any,
        split: pd.DataFrame,
        columns: list[str],
        *,
        strict_cutoff: bool = False,
    ) -> tuple[pd.DataFrame, list[str]]:
        del db, task, split, columns, strict_cutoff
        output = features.copy()
        # Every training row's source was null; the attach path stringified it.
        output["notes__raw_text"] = ["nan", "None", "<NA>"][: len(output)]
        return output, ["notes__raw_text"]

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", fake_features)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: ["notes"])
    monkeypatch.setattr(model_module, "attach_anchor_text", fake_attach)
    task = SimpleNamespace(
        target_col="target",
        time_col=None,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    model = NoriRel(dict(model_module.DEFAULT_CONFIG))

    model.fit(task, object(), train, None, seed=0)

    assert model._columns == ["value"]
    assert model._text_columns == []


def test_import_does_not_load_optional_nori_dependency() -> None:
    code = (
        "import sys; import relarena.models.nori_rel; "
        "assert 'synthefy_nori' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("task_type", "output_type", "expected"),
    [
        (TaskType.REGRESSION, "median", [-0.25, 0.5, 1.25]),
        (TaskType.BINARY_CLASSIFICATION, "mean", [0.0, 0.5, 1.0]),
    ],
)
def test_prediction_contract(
    monkeypatch: pytest.MonkeyPatch,
    task_type: TaskType,
    output_type: str,
    expected: list[float],
) -> None:
    feature_calls: list[dict[str, Any]] = []

    def fake_features(
        task: Any,
        db: Any,
        table: Any,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db
        feature_calls.append(kwargs)
        rows = len(table.df)
        frame = pd.DataFrame({"value": np.arange(rows), "group": ["x"] * rows})
        return frame, ["group"]

    def fake_attach(
        features: pd.DataFrame,
        db: Any,
        task: Any,
        split: pd.DataFrame,
        columns: list[str],
        *,
        strict_cutoff: bool = False,
    ) -> tuple[pd.DataFrame, list[str]]:
        del db, task, split
        assert columns == ["description"]
        assert strict_cutoff is False
        output = features.copy()
        output["description__raw_text"] = ["text"] * len(output)
        return output, ["description__raw_text"]

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", fake_features)
    monkeypatch.setattr(
        model_module,
        "anchor_text_columns",
        lambda db, task: ["description"],
    )
    monkeypatch.setattr(model_module, "attach_anchor_text", fake_attach)

    task = SimpleNamespace(
        target_col="target",
        time_col="timestamp",
        task_type=task_type,
    )
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(3)))
    model = NoriRel(dict(model_module.DEFAULT_CONFIG))

    model.fit(task, object(), train, None, seed=7)
    prediction = model.predict(task, object(), query)

    assert model._model.kwargs == {
        "model_path": "/nori-30m.pt",
        "categorical_columns": ["group"],
        "memory_policy": model_module._memory_policy(),
        "large_context_policy": "random",
        "large_context_threshold": 4,
        "large_context_seed": 7,
        "text_columns": ["description__raw_text"],
        "svd_dim": model_module.TEXT_SVD_DIM,
        "embedder": model_module.TEXT_EMBEDDER,
    }
    assert model._model.fit_rows == 3
    assert model._model.output_type == output_type
    assert [call["depth"] for call in feature_calls] == [2, 2]
    # The shared cache is keyed on max_depth, so the entry must build the same
    # deepest matrix as RDBLearn and the default warmer, then slice to depth 2.
    assert [call["max_depth"] for call in feature_calls] == [
        model_module.DFS_MAX_DEPTH
    ] * 2
    np.testing.assert_array_equal(prediction, expected)


def test_public_checkpoint_is_pinned_and_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "nori.pt"
    path.write_bytes(b"released weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = 0

    def fake_download() -> str:
        nonlocal calls
        calls += 1
        return str(path)

    monkeypatch.setattr(model_module, "_download_checkpoint", fake_download)
    monkeypatch.setattr(model_module, "CHECKPOINT_SHA256", digest)
    model_module._checkpoint_path.cache_clear()
    model_module._sha256.cache_clear()

    assert model_module._checkpoint_path() == str(path)
    assert calls == 1

    path.write_bytes(b"different weights")
    model_module._checkpoint_path.cache_clear()
    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        model_module._checkpoint_path()


def test_large_context_uses_seeded_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_features(
        task: Any,
        db: Any,
        table: Any,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, list[str]]:
        del task, db, kwargs
        rows = len(table.df)
        return pd.DataFrame({"a": range(rows), "b": range(rows)}), []

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "build_dfs_features", fake_features)
    monkeypatch.setattr(model_module, "CONTEXT_ELEMENTS_BUDGET", 3)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])

    task = SimpleNamespace(
        target_col="target", time_col=None, task_type=TaskType.REGRESSION
    )
    train = SimpleNamespace(df=pd.DataFrame({"target": [1.0, 2.0, 3.0]}))
    model = NoriRel(dict(model_module.DEFAULT_CONFIG))
    model.fit(task, object(), train, None, seed=0)

    assert model._model.kwargs["large_context_threshold"] == 1
    assert model._model.kwargs["large_context_policy"] == "random"


def test_text_context_is_seeded_and_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_module, "MAX_CONTEXT_ROWS", 3)
    expected = np.random.default_rng(7).permutation(8)[:3]

    np.testing.assert_array_equal(
        model_module._text_context_indices(8, 7),
        expected,
    )
    np.testing.assert_array_equal(
        model_module._text_context_indices(3, 7),
        [0, 1, 2],
    )


def test_gpu_budget_disables_lossy_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(model_module.GPU_BUDGET_GB_ENV, "20")

    assert model_module._memory_policy() == {
        "allow_quantization": False,
        "allow_subsample": False,
        "context_row_chunk": model_module.CONTEXT_ROW_CHUNK,
        "elements_budget": model_module.CONTEXT_ELEMENTS_BUDGET,
        "gpu_budget_absolute_gb": 20.0,
    }


def test_reported_config_is_fixed() -> None:
    model = NoriRel({"max_depth": 3, "context_policy": "random"})
    task = SimpleNamespace(
        target_col="target", time_col=None, task_type=TaskType.REGRESSION
    )
    train = SimpleNamespace(df=pd.DataFrame({"target": [1.0]}))

    with pytest.raises(ValueError, match="requires depth 2, random context"):
        model.fit(task, object(), train, None, seed=0)


def test_prefill_is_unchunked_by_default() -> None:
    """Pinning a prefill chunk cost 10-14% and bought nothing.

    Unchunked is the memory policy's own default and the value it falls back to
    chunking from after an out-of-memory error, so pinning a number also spent
    that escalation in advance. Bit-exact either way, which is why this is a
    default rather than an option.
    """
    assert model_module.CONTEXT_ROW_CHUNK is None
    assert model_module._memory_policy()["context_row_chunk"] is None


def test_fixed_noridfs_backend_uses_the_native_featurizer_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeFeaturizer:
        def __init__(self, config: Any, *, with_target_history: bool) -> None:
            self.config = config
            self.with_target_history = with_target_history

        def fit(self, task: Any, db: Any, table: Any, *, history_table: Any) -> Any:
            calls.append("fit")
            assert history_table is table
            return self

        def transform(self, task: Any, db: Any, table: Any) -> Any:
            calls.append("transform")
            rows = len(table.df)
            return pd.DataFrame({"value": np.arange(rows), "kind": ["a"] * rows}), [
                "kind"
            ]

    def fail_fastdfs(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("FastDFS must not run under the NoriDFS backend")

    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "RelBenchNativeFeaturizer", FakeFeaturizer)
    monkeypatch.setattr(model_module, "build_dfs_features", fail_fastdfs)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])

    task = SimpleNamespace(
        target_col="target", time_col="timestamp", task_type=TaskType.REGRESSION
    )
    train = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0, 0.0]}))
    query = SimpleNamespace(df=pd.DataFrame(index=range(3)))
    model = model_module.NoriRelNoriDFS(dict(model_module.NORIDFS_FIXED_CONFIG))

    model.fit(task, object(), train, None, seed=7)
    prediction = model.predict(task, object(), query)

    assert model.name == "nori-rel-noridfs"
    assert model.supported_task_types == frozenset({TaskType.REGRESSION})
    assert model.config["feature_backend"] == model_module.NORIDFS_BACKEND
    assert calls == ["fit", "transform", "transform"]
    featurizer = model._noridfs
    assert featurizer.with_target_history is True
    assert featurizer.config.max_features == 48
    assert featurizer.config.max_direct_features == 1
    assert featurizer.config.window_multipliers == (1, 4, 16)
    assert model._model.kwargs["categorical_columns"] == ["kind"]
    assert model._model.output_type == "median"
    np.testing.assert_array_equal(prediction, [-0.25, 0.5, 1.25])


def test_noridfs_is_not_registered_and_rejects_unknown_backend() -> None:
    assert "nori-rel-noridfs" not in registry
    with pytest.raises(ValueError, match="feature_backend"):
        model_module.NoriRel(
            {**model_module.DEFAULT_CONFIG, "feature_backend": "x"}
        ).fit(
            SimpleNamespace(
                target_col="t", time_col=None, task_type=TaskType.REGRESSION
            ),
            object(),
            SimpleNamespace(df=pd.DataFrame({"t": [0.0]})),
            None,
            seed=0,
        )


def _relbench_fixture() -> tuple[Database, SimpleNamespace, Table, Table, Table]:
    users = Table(
        pd.DataFrame({"user_id": [1, 2], "age": [20.0, 30.0]}),
        fkey_col_to_pkey_table={},
        pkey_col="user_id",
    )
    events = Table(
        pd.DataFrame(
            {
                "event_id": [10, 11, 12],
                "user_id": [1, 1, 2],
                "event_time": pd.to_datetime(
                    ["2026-01-01", "2026-01-03", "2026-01-02"]
                ),
                "value": [1.0, 2.0, 3.0],
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users"},
        pkey_col="event_id",
        time_col="event_time",
    )
    database = Database({"users": users, "events": events})
    task = SimpleNamespace(
        entity_table="users",
        entity_col="user_id",
        time_col="cutoff",
        target_col="target",
        timedelta=pd.Timedelta("7 days"),
        task_type=TaskType.REGRESSION,
    )

    def split(users: list[int], days: list[str], targets: list[float] | None) -> Table:
        frame = pd.DataFrame({"user_id": users, "cutoff": pd.to_datetime(days)})
        if targets is not None:
            frame["target"] = targets
        return Table(
            frame, fkey_col_to_pkey_table={"user_id": "users"}, time_col="cutoff"
        )

    train = split(
        [1, 2, 1], ["2026-01-02", "2026-01-03", "2026-01-04"], [1.0, 2.0, 3.0]
    )
    validation = split([2, 1], ["2026-01-04", "2026-01-05"], [4.0, 5.0])
    test = split([1, 2], ["2026-01-06", "2026-01-06"], None)
    return database, task, train, validation, test


def test_tuned_noridfs_arms_run_inner_and_outer_lifecycle_without_fastdfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, task, train, validation, test = _relbench_fixture()
    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    monkeypatch.setattr(
        model_module,
        "build_dfs_features",
        lambda *args, **kwargs: pytest.fail("noridfs model called FastDFS"),
    )

    inner = model_module.NoriRelNoriDFS(dict(model_module.NORIDFS_DIRECT_CONFIG))
    inner.fit(task, database, train, validation, seed=0)
    validation_prediction = inner.predict(task, database, validation)
    plan = inner._noridfs._featurizer.plan
    assert plan.config.max_direct_categorical_cardinality == 32
    assert plan.config.max_features == 87

    full = Table(
        pd.concat([train.df, validation.df], ignore_index=True),
        fkey_col_to_pkey_table={"user_id": "users"},
        time_col="cutoff",
    )
    outer = model_module.NoriRelNoriDFS(dict(model_module.NORIDFS_MISSING_CONFIG))
    outer.fit(task, database, full, None, seed=0)
    test_prediction = outer.predict(task, database, test)

    assert validation_prediction.shape == (2,)
    assert test_prediction.shape == (2,)
    config = outer._noridfs._featurizer.plan.config
    assert config.max_features == 87
    assert config.include_missing_in_modes and config.include_missing_fractions
    assert outer._history is not None and outer._history.n_lags == 5
    # Only the shallow lags carry values in this fixture. A lag that is entirely
    # missing in the training context is dropped, not handed to Nori empty.
    assert "target_lag1" in outer._columns
    assert "target_lag5_age_days" not in outer._columns
    options = outer._model.kwargs
    assert options["large_context_policy"] is model_module.cache_safe_random_window
    assert options["memory_policy"] == model_module.NORIDFS_MEMORY_POLICY
    assert options["discretize"] == "snap-median"
    assert options["categorical_levels"] == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert outer._model.output_type == "mean"
    assert outer.memory_audit_["memory_rung_actual"] == "no_cache"
    assert list(outer._model.fit_features) == list(outer._model.predict_features)


def test_fixed_noridfs_config_keeps_the_baseline_inference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, task, train, _, test = _relbench_fixture()
    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])

    model = model_module.NoriRelNoriDFS(dict(model_module.NORIDFS_FIXED_CONFIG))
    model.fit(task, database, train, None, seed=0)
    model.predict(task, database, test)

    options = model._model.kwargs
    assert model._history is None
    assert options["memory_policy"] == model_module._memory_policy()
    assert "discretize" not in options
    assert model._model.output_type == "median"
    assert not hasattr(model, "memory_audit_")


def test_fastdfs_backend_rejects_noridfs_keys() -> None:
    task = SimpleNamespace(target_col="t", time_col=None, task_type=TaskType.REGRESSION)
    train = SimpleNamespace(df=pd.DataFrame({"t": [0.0]}))

    with pytest.raises(ValueError, match="need the noridfs backend"):
        NoriRel({**model_module.DEFAULT_CONFIG, "train_profile": "v1"}).fit(
            task, object(), train, None, seed=0
        )


def test_candidate_search_space_is_the_three_arm_grid() -> None:
    space = model_module.NORIDFS_SPACE

    assert space.default_overrides == model_module.NORIDFS_LEAN_CONFIG
    assert space.fixed_grid == model_module.NORIDFS_GRID
    assert len(model_module.NORIDFS_GRID) == 3
    assert model_module.NoriRelNoriDFS().config == model_module.NORIDFS_LEAN_CONFIG


def test_noridfs_fit_keeps_reservoir_features_and_targets_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFeaturizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def fit(self, task: Any, db: Any, table: Table, *, history_table: Any) -> Any:
            del task, db, history_table
            self.fit_rows = table.df["row_id"].tolist()
            return self

        def transform(self, task: Any, db: Any, table: Table) -> Any:
            del task, db
            return table.df[["row_id"]].reset_index(drop=True), []

    from relarena.models.nori_rel import context as context_module

    monkeypatch.setattr(model_module, "MAX_CONTEXT_ROWS", 4)
    monkeypatch.setattr(context_module, "MAX_CONTEXT_ROWS", 4)
    monkeypatch.setattr(context_module, "TEMPORAL_RESERVOIR_RECENT_ROWS", 2)
    monkeypatch.setattr(model_module, "RelBenchNativeFeaturizer", FakeFeaturizer)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    _install_fake_nori(monkeypatch)

    frame = pd.DataFrame(
        {
            "row_id": np.arange(8),
            "entity": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "cutoff": pd.date_range("2026-01-01", periods=8),
            "target": np.tile([1.0, 2.0], 4),
        }
    )
    table = Table(frame, fkey_col_to_pkey_table={}, time_col="cutoff")
    task = SimpleNamespace(
        entity_col="entity",
        time_col="cutoff",
        target_col="target",
        task_type=TaskType.REGRESSION,
    )
    expected = context_module.temporal_context_indices(task, table, 7)
    rare_row = next(row for row in range(len(frame)) if row not in set(expected))
    table.df.loc[rare_row, "target"] = 99.0
    model = model_module.NoriRelNoriDFS(
        {**model_module.NORIDFS_LEAN_CONFIG, "train_profile": "none"}
    )

    model.fit(task, object(), table, None, seed=7)

    expected_rows = table.df.iloc[expected].reset_index(drop=True)
    assert model._noridfs.fit_rows == expected_rows["row_id"].tolist()
    np.testing.assert_array_equal(
        model._model.fit_features["row_id"], expected_rows["row_id"]
    )
    np.testing.assert_array_equal(model._model.target, expected_rows["target"])
    # The lattice reads the full training split, not the reservoir.
    assert model._model.kwargs["categorical_levels"] == (1.0, 2.0, 99.0)


def test_plain_loop_fallback_is_audited_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlainLoopNoriRegressor(FakeNoriRegressor):
        def predict(self, features: pd.DataFrame, *, output_type: str) -> np.ndarray:
            warnings.warn(
                "Nori fell back to the plain chunked loop: forced test",
                RuntimeWarning,
                stacklevel=2,
            )
            self.memory_report_["rung"] = "plain_loop"
            return super().predict(features, output_type=output_type)

    database, task, train, _, test = _relbench_fixture()
    monkeypatch.setattr(
        model_module, "_load_nori", lambda: (PlainLoopNoriRegressor, nullcontext)
    )
    monkeypatch.setattr(model_module, "_checkpoint_path", lambda: "/nori-30m.pt")
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])
    model = model_module.NoriRelNoriDFS()
    model.fit(task, database, train, None, seed=0)

    with pytest.warns(RuntimeWarning, match="plain chunked loop"):
        prediction = model.predict(task, database, test)

    assert prediction.shape == (2,)
    assert model.memory_audit_["memory_rung_actual"] == "plain_loop"


@pytest.mark.parametrize(
    ("targets", "expected_lags", "expected_decoder"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 5, "median"),
        ([0.0, 1.0, 1.0, 0.0, 1.0], 0, "median"),
    ],
    ids=["ordinary-regression", "endpoint-rich-rate"],
)
def test_endpoint_rich_targets_skip_lags_and_keep_the_median(
    monkeypatch: pytest.MonkeyPatch,
    targets: list[float],
    expected_lags: int,
    expected_decoder: str,
) -> None:
    database, task, train, validation, _ = _relbench_fixture()
    full = Table(
        pd.concat([train.df, validation.df], ignore_index=True).assign(target=targets),
        fkey_col_to_pkey_table={"user_id": "users"},
        time_col="cutoff",
    )
    _install_fake_nori(monkeypatch)
    monkeypatch.setattr(model_module, "anchor_text_columns", lambda db, task: [])

    # A 0/1 rate on five rows is also an integer lattice; turn that rule off so
    # the endpoint route is what decides here.
    model = model_module.NoriRelNoriDFS(
        {**model_module.NORIDFS_LEAN_CONFIG, "noridfs_target_lattice": False}
    )
    model.fit(task, database, full, None, seed=0)
    model.predict(task, database, validation)

    lags = 0 if model._history is None else model._history.n_lags
    assert lags == expected_lags
    assert model._model.output_type == expected_decoder
