"""Load the packaged RelFlow schemas, feature queries, and training settings."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_recipe(dataset: str, task: str) -> dict[str, Any]:
    """Return a fresh configuration for one supported RelBench entity task.

    The optional RelFlow runtime is not imported while loading configurations.
    """
    import yaml

    resources = files(__package__).joinpath("recipes")
    recipes = json.loads(resources.joinpath("manifest.json").read_text())
    name = f"{dataset}/{task}"
    if name not in recipes:
        raise ValueError(
            f"RelFlow has no recipe for {name!r}; supported tasks: "
            + ", ".join(sorted(recipes))
        )
    recipe = recipes[name]
    directory = resources.joinpath(dataset, task)
    recipe["model"] = yaml.safe_load(directory.joinpath("model.yaml").read_text())
    recipe["query"] = directory.joinpath("observations.sql").read_text()
    return recipe
