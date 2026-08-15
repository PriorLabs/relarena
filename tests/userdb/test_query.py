"""Tests for the PredictiveQuery façade (entity-id translation)."""

from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import indent
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from relarena.cache import CacheConfig
from relarena.identity import RunIdentity
from relarena.userdb import relbench_v1_spec, relbench_v1_tasks
from relarena.userdb.query import PredictiveQuery, PredictiveQuerySpec

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_DB_YAML = "drivers:\n  pkey: driverId\n"
_TASK_YAML = (
    "database: db.yaml\n"
    "entity_table: drivers\n"
    "entity_col: driverId\n"
    "time_col: date\n"
    "target_col: y\n"
    "task_type: binary_classification\n"
    "timedelta: 30 days\n"
    "val_timestamp: '2005-01-01'\n"
    "test_timestamp: '2005-01-31'\n"
    "query: SELECT 1\n"
)


def _write_pair(tmp_path: Path, *, task: str = _TASK_YAML, db: str = _DB_YAML) -> str:
    """Write a task.yaml plus its sibling db.yaml; return the task file path."""
    (tmp_path / "db.yaml").write_text(db)
    task_path = tmp_path / "task.yaml"
    task_path.write_text(task)
    return str(task_path)


def _id_map() -> pd.Series:
    """Original entity id -> reindexed 0..n-1 id."""
    return pd.Series(data=[0, 1, 2], index=["s_a", "s_b", "s_c"])


def test__to_internal_ids__all_selector__passes_through() -> None:
    assert PredictiveQuery._to_internal_ids("all", _id_map()) == "all"


def test__to_internal_ids__no_map__passes_through() -> None:
    # Native RelBench datasets have no id map; ids stay as given.
    assert PredictiveQuery._to_internal_ids(["s_a"], None) == ["s_a"]


def test__to_internal_ids__original_ids__mapped_to_indices() -> None:
    assert PredictiveQuery._to_internal_ids(["s_c", "s_a"], _id_map()) == [2, 0]


def test__to_internal_ids__unknown_ids__dropped_with_warning() -> None:
    with pytest.warns(UserWarning, match="absent from the database"):
        out = PredictiveQuery._to_internal_ids(["s_a", "nope"], _id_map())
    assert out == [0]


def test__to_internal_ids__scalar__raises() -> None:
    with pytest.raises(ValueError, match="got scalar"):
        PredictiveQuery._to_internal_ids(5, _id_map())


def test__from_yaml__task_and_database__loaded_and_composed(tmp_path: Path) -> None:
    """from_yaml loads the task file and its referenced sibling database YAML."""
    spec = PredictiveQuerySpec.from_yaml(_write_pair(tmp_path))
    assert set(spec.database.tables) == {"drivers"}
    assert spec.task.entity_col == "driverId"
    assert spec.task.val_timestamp == pd.Timestamp("2005-01-01")


def test__from_yaml__null_entities__raises(tmp_path: Path) -> None:
    """An explicit `entities: null` fails schema validation on load."""
    task = _TASK_YAML + "entities: null\n"
    with pytest.raises(ValueError, match="Invalid task YAML"):
        PredictiveQuerySpec.from_yaml(_write_pair(tmp_path, task=task))


def test__from_yaml__unquoted_timestamps__accepted(tmp_path: Path) -> None:
    """Unquoted YAML timestamps (parsed as date) pass validation and normalize."""
    task = _TASK_YAML.replace("'2005-01-01'", "2005-01-01").replace(
        "'2005-01-31'", "2005-01-31"
    )
    spec = PredictiveQuerySpec.from_yaml(_write_pair(tmp_path, task=task))
    assert spec.task.val_timestamp == pd.Timestamp("2005-01-01")


def test__fit__constant_global_model__predicts_and_computes_test_labels(
    tmp_path: Path,
) -> None:
    """The full fit -> predict flow can materialize labels for its test cohort."""
    months = pd.date_range("2004-01-15", "2005-06-15", freq="30D")
    events = pd.DataFrame(
        {
            "eventId": range(2 * len(months)),
            "driverId": [0] * len(months) + [1] * len(months),
            "date": list(months) * 2,
        }
    )
    pd.DataFrame({"driverId": [0, 1, 2, 3]}).to_parquet(tmp_path / "drivers.parquet")
    events.to_parquet(tmp_path / "events.parquet")
    db = _DB_YAML + (
        "events:\n  pkey: eventId\n  time_col: date\n  fkeys:\n    driverId: drivers\n"
    )
    query = (
        "SELECT t.timestamp AS date, d.driverId AS driverId,\n"
        "  CAST(EXISTS (\n"
        "    SELECT 1 FROM events e WHERE e.driverId = d.driverId\n"
        "      AND e.date > t.timestamp\n"
        "      AND e.date <= t.timestamp + INTERVAL '{timedelta}'\n"
        "  ) AS INTEGER) AS y\n"
        "FROM timestamp_df t CROSS JOIN drivers d\n"
        "WHERE EXISTS (\n"
        "  SELECT 1 FROM events h WHERE h.driverId = d.driverId\n"
        "    AND h.date > t.timestamp - INTERVAL '{timedelta}'\n"
        "    AND h.date <= t.timestamp\n"
        ")\n"
    )
    task = (
        _TASK_YAML.replace("query: SELECT 1\n", f"query: |\n{indent(query, '  ')}")
        .replace("'2005-01-01'", "'2004-10-01'")
        .replace("'2005-01-31'", "'2004-12-01'")
    )
    spec = PredictiveQuerySpec.from_yaml(
        _write_pair(tmp_path, task=task, db=db), data_dir=str(tmp_path)
    )

    query = PredictiveQuery(spec, data_version="tiny-v1").fit(
        "constant-global", n_trials=0, cache_dir=tmp_path / "cache"
    )
    preds = query.predict()
    labels = query.compute_test_labels()

    assert sorted(preds["driverId"]) == [0, 1, 2, 3]
    assert preds["date"].unique().tolist() == [pd.Timestamp("2004-12-01")]
    assert "y_pred" in preds.columns
    assert list(labels.columns) == ["date", "driverId", "y"]
    assert sorted(labels["driverId"]) == [0, 1]
    scored = labels.merge(
        preds,
        on=["date", "driverId"],
        how="left",
        validate="one_to_one",
    )
    assert scored["y_pred"].notna().all()
    assert query._model.cache.directory == tmp_path / "cache"
    assert query._model.cache.on_miss == "fill"
    assert query._model.run_identity.data_version == "tiny-v1"
    assert query._model.run_identity.phase == "predict"

    with pytest.raises(ValueError, match="require data through"):
        query.compute_test_labels(data_end_timestamp="2004-12-15")


def _schema_only_query(*, data_version: str | None = None) -> PredictiveQuery:
    query = PredictiveQuery.__new__(PredictiveQuery)
    query._identity = RunIdentity(
        "user", "schema", "drivers-dnf", "task", data_version=data_version
    )
    query._warned_schema_only_cache = False
    return query


def test__precompute_cache__delegates_to_dfs_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = _schema_only_query(data_version="v1")
    query._source = Mock()
    warm = Mock()
    monkeypatch.setattr("relarena.featurization.warm_cache.warm_dfs_cache", warm)

    assert query.precompute_cache(tmp_path) == tmp_path

    warm.assert_called_once_with(query._source, CacheConfig(tmp_path, "fill"))


def test__warn_schema_only_cache__persistent_store__warns_once(
    tmp_path: Path,
) -> None:
    query = _schema_only_query()
    cache = CacheConfig(tmp_path, "fill")

    with pytest.warns(UserWarning, match="no data_version") as recorded:
        query._warn_schema_only_cache(cache)
        query._warn_schema_only_cache(cache)

    assert len(recorded) == 1


@pytest.mark.parametrize(
    "query,cache",
    [
        (_schema_only_query(), CacheConfig(None, "compute")),
        (_schema_only_query(data_version="v1"), CacheConfig(Path("cache"), "fill")),
    ],
)
def test__warn_schema_only_cache__unambiguous_use__does_not_warn(
    query: PredictiveQuery, cache: CacheConfig
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        query._warn_schema_only_cache(cache)


def test__predict__anchor_after_test_cutoff__warns_about_frozen_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _schema_only_query(data_version="v1")
    frozen_db = Mock()
    test_timestamp = pd.Timestamp("2020-01-01")
    query._source = SimpleNamespace(
        _db=frozen_db,
        _dataset=SimpleNamespace(test_timestamp=test_timestamp),
    )
    query.task = SimpleNamespace(entity_table="drivers", entity_col="driverId")
    query._model = SimpleNamespace(cache=None, run_identity=None)
    query._cache = CacheConfig(directory=None, on_miss="compute")
    query._at_timestamp = pd.Timestamp("2020-02-01")
    query._entities = "all"
    predict_at = Mock(return_value=pd.DataFrame({"driverId": [], "y_pred": []}))
    monkeypatch.setattr("relarena.userdb.query.predict_at", predict_at)

    with pytest.warns(UserWarning, match="feature database remains frozen"):
        query.predict()

    predict_at.assert_called_once_with(
        query._model,
        query.task,
        frozen_db,
        pd.Timestamp("2020-02-01"),
        "all",
    )


@pytest.mark.parametrize("dataset,task", relbench_v1_tasks())
def test__relbench_v1_spec__task_and_db_conform_to_schema(
    dataset: str, task: str
) -> None:
    """Every shipped RelBench-v1 spec loads: its task and db.yaml both validate."""
    relbench_v1_spec(dataset, task, data_dir="data")  # validates task + database


def test__from_yaml__olist_example__task_and_db_conform_to_schema() -> None:
    """The worked Olist example (task file + its database YAML) loads and validates."""
    PredictiveQuerySpec.from_yaml(
        str(_EXAMPLES / "olist_seller_churn.yaml"), data_dir="data"
    )
