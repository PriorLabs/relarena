from __future__ import annotations

import json
import subprocess
import sys
from importlib.resources import files

import pytest

from relarena.models.relflow.config import load_recipe

DATASETS = {
    "rel-amazon": ("item-churn", "item-ltv", "user-churn", "user-ltv"),
    "rel-avito": ("ad-ctr", "user-clicks", "user-visits"),
    "rel-event": ("user-attendance", "user-ignore", "user-repeat"),
    "rel-f1": ("driver-dnf", "driver-position", "driver-top3"),
    "rel-hm": ("item-sales", "user-churn"),
    "rel-stack": ("post-votes", "user-badge", "user-engagement"),
    "rel-trial": ("site-success", "study-adverse", "study-outcome"),
}
TASKS = [(dataset, task) for dataset, tasks in DATASETS.items() for task in tasks]


@pytest.mark.parametrize(("dataset", "task"), TASKS)
def test_recipe_contract(dataset: str, task: str) -> None:
    recipe = load_recipe(dataset, task)
    assert recipe["model"]["fields"]["target"]["mask"] is True
    assert recipe["contract"]["task_type"] in {
        "binary_classification",
        "regression",
    }
    assert recipe["contract"]["entity_table"] in recipe["contract"]["sources"]
    assert recipe["training"]["max_epochs"] == 40
    assert recipe["training"]["max_steps"] > 0
    assert recipe["training"]["min_full_passes"] == 2
    assert all(
        parameter in recipe["query"] for parameter in ("$anchors", "$start", "$stop")
    )
    assert '"__row_position"' in recipe["query"]
    assert '"__target"' not in recipe["query"]


def test_all_entity_tasks_are_available() -> None:
    resources = files("relarena.models.relflow").joinpath("recipes")
    recipes = json.loads(resources.joinpath("manifest.json").read_text())
    assert set(recipes) == {f"{dataset}/{task}" for dataset, task in TASKS}
    assert len(recipes) == 21


def test_dnf_supervised_target() -> None:
    model = load_recipe("rel-f1", "driver-dnf")["model"]
    assert model["fields"]["target"] == {
        "type": "boolean",
        "mask": True,
        "pooling": "mean",
        "n_linear": 1,
        "n_heads": 4,
        "dropout": 0.2,
    }
    assert model["fields"]["results"]["length"] == 128
    assert model["fields"]["results"]["reduction"] == {"type": "mean"}


def test_loading_returns_independent_configuration() -> None:
    first = load_recipe("rel-f1", "driver-top3")
    first["model"]["fields"].clear()
    first["training"]["max_steps"] = 0
    second = load_recipe("rel-f1", "driver-top3")
    assert second["model"]["fields"]
    assert second["training"]["max_steps"] > 0


def test_unsupported_task() -> None:
    with pytest.raises(ValueError, match="has no recipe.*unknown-task"):
        load_recipe("rel-f1", "unknown-task")


def test_config_does_not_import_optional_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import relarena.models.relflow.config; "
            "assert 'relflow' not in sys.modules",
        ],
        check=True,
    )


def test_known_empty_column_whitelist_is_task_specific() -> None:
    replacements = {}
    for dataset, task in TASKS:
        recipe = load_recipe(dataset, task)
        for table, source in recipe["contract"]["sources"].items():
            if source.get("nullable_columns"):
                replacements[(dataset, task, table)] = source["nullable_columns"]
    expected = {"limitations_and_caveats": "VARCHAR"}
    assert replacements == {
        ("rel-trial", "study-adverse", "studies"): expected,
        ("rel-trial", "study-outcome", "studies"): expected,
    }
