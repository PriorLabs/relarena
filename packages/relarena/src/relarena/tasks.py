"""Named RelBench benchmark tasks and dataset selection."""

from __future__ import annotations

from typing import NamedTuple, Sequence

from relbench.base import TaskType

from relarena.core.tasks import ENTITY_TASK_TYPES as ENTITY_TASK_TYPES

#: The seven original RelBench (v1) datasets.
#:
#: NB: "v1" here refers to the *dataset/benchmark generation*, not the package
#: version. We pin the `relbench` **v2** package (see `pyproject.toml`) but
#: currently only run against this original v1 dataset set; the newer datasets
#: shipped with relbench v2 are intentionally out of scope for now. The same
#: "v1" sense is used in the recorded checksums (`relarena.checksums`) and
#: `workflows/record_checksums.py`.
RELBENCH_V1_DATASETS: tuple[str, ...] = (
    "rel-amazon",
    "rel-avito",
    "rel-event",
    "rel-f1",
    "rel-hm",
    "rel-stack",
    "rel-trial",
)


class TaskSpec(NamedTuple):
    """A discovered task: which dataset, its name, and its type."""

    dataset: str
    task: str
    task_type: TaskType


def list_entity_tasks(
    datasets: Sequence[str] | None = None,
    *,
    task_types: frozenset[TaskType] = ENTITY_TASK_TYPES,
) -> list[TaskSpec]:
    """Discover in-scope entity tasks from RelBench's registry **without downloading**.

    Reads `task_type` straight off the registered task classes (a class
    attribute), so no Database is built. Excludes:
      * non-entity tasks (link prediction);
      * `AutoCompleteTask` (it subclasses `EntityTask` but is out of scope);
      * tasks whose class-level `task_type` is unset or not in `task_types`.

    Args:
        datasets: dataset names to scan; defaults to all registered datasets.
        task_types: which task types to keep (default: entity classification +
            regression).
    """
    from relbench.base import EntityTask
    from relbench.tasks import task_registry

    try:
        from relbench.base.task_autocomplete import AutoCompleteTask
    except Exception:  # pragma: no cover - defensive if relbench moves it
        AutoCompleteTask = None

    names = list(task_registry) if datasets is None else datasets
    specs: list[TaskSpec] = []
    for ds in names:
        for task_name, entry in task_registry.get(ds, {}).items():
            cls = entry[0] if isinstance(entry, tuple) else entry
            if not (isinstance(cls, type) and issubclass(cls, EntityTask)):
                continue
            if AutoCompleteTask is not None and issubclass(cls, AutoCompleteTask):
                continue
            task_type = getattr(cls, "task_type", None)
            if task_type not in task_types:
                continue
            specs.append(TaskSpec(ds, task_name, task_type))
    return specs
