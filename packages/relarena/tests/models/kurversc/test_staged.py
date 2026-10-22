from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from relbench.base import TaskType

from relarena.models.kurversc import KurveRSC20261022System, KurveRSCSystem
from relarena_core import CacheConfig, RunIdentity
from relarena_core.registry import registry


def test__staged__has_distinct_versioned_registration_and_latest_alias() -> None:
    assert registry.get("kurversc") is KurveRSCSystem
    assert registry.get("kurversc-2026-10-22") is KurveRSC20261022System
    assert registry.get("kurversc-latest") is KurveRSC20261022System
    assert registry.kind("kurversc-latest") == "system"
    with pytest.raises(TypeError, match="no harness search space"):
        registry.search_space("kurversc-latest")
    assert KurveRSC20261022System.supported_task_types == frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )


@pytest.mark.parametrize("seed", [1, 7, -1])
def test__staged__unsupported_seed_fails_before_optional_import(seed: int) -> None:
    with pytest.raises(ValueError, match="deployment seed 0"):
        KurveRSC20261022System().run(
            SimpleNamespace(),
            inner_split=SimpleNamespace(),
            outer_split=SimpleNamespace(),
            seed=seed,
        )


@pytest.mark.parametrize("identified", [True, False])
def test__staged__forwards_original_splits_budget_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identified: bool
) -> None:
    train = object()
    query = pd.DataFrame({"entity": [7, 3], "at": pd.to_datetime(["2020-02-01"] * 2)})
    inner = SimpleNamespace(train_table=train)
    outer = SimpleNamespace(eval_table=SimpleNamespace(df=query))
    task = SimpleNamespace(target_col="response")
    identity = (
        RunIdentity("new-schema", "source-v1", "new-task", "task-v1")
        if identified
        else None
    )
    output = tmp_path / "owned-complete-run"
    calls = []

    def run_protocol_task(actual_task: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        assert actual_task is task
        assert kwargs["inner_split"] is inner and kwargs["outer_split"] is outer
        assert "response" not in kwargs["outer_split"].eval_table.df
        assert kwargs["time_limit"] == 600
        assert kwargs["seed"] == 0
        assert kwargs["cache_directory"] == tmp_path
        assert kwargs["run_identity"] is identity
        return SimpleNamespace(
            output=output, predictions=pd.DataFrame({"prediction": [0.8, 0.2]})
        )

    monkeypatch.setitem(
        sys.modules,
        "kurversc.protocol_runner",
        SimpleNamespace(run_protocol_task=run_protocol_task),
    )
    system = KurveRSC20261022System(
        cache=CacheConfig(directory=tmp_path, on_miss="compute"), run_identity=identity
    )
    prediction = system.run(
        task, inner_split=inner, outer_split=outer, seed=0, time_limit=600
    )
    np.testing.assert_array_equal(prediction, [0.8, 0.2])
    assert prediction.dtype == float and prediction.shape == (len(query),)
    assert system.complete_pipeline_output == output
    assert calls[0]["dataset_name"] == ("new-schema" if identified else "database")
    assert calls[0]["task_name"] == ("new-task" if identified else "prediction")


def test__staged__registration_does_not_import_optional_pipeline() -> None:
    code = """
import sys
from relarena.models.kurversc import KurveRSC20261022System
assert 'kurversc.protocol_runner' not in sys.modules
assert 'kurversc.staged' not in sys.modules
assert KurveRSC20261022System.name == 'kurversc-2026-10-22'
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test__staged__pipeline_errors_are_not_replaced_with_native_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Incomplete full-fidelity stage")

    monkeypatch.setitem(
        sys.modules,
        "kurversc.protocol_runner",
        SimpleNamespace(run_protocol_task=failed),
    )
    with pytest.raises(RuntimeError, match="Incomplete full-fidelity"):
        KurveRSC20261022System().run(
            SimpleNamespace(),
            inner_split=SimpleNamespace(),
            outer_split=SimpleNamespace(),
            seed=0,
        )
