"""Supported relational prediction task types."""

from relbench.base import TaskType

#: Entity (node-level) task types RelArena supports. Excludes:
#:   * `LINK_PREDICTION` — recommendation, out of scope;
#:   * `MULTILABEL_CLASSIFICATION` — RelBench has no entity multilabel task
#:     (its sole multilabel task is a TGB node-property ranking `BaseTask`).
#: Easy to re-add if a real entity multilabel task appears.
ENTITY_TASK_TYPES: frozenset[TaskType] = frozenset(
    {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
    }
)

# The rest of the codebase assumes exactly these two task types; guard against
# silently widening scope without revisiting those call sites.
assert ENTITY_TASK_TYPES == {
    TaskType.REGRESSION,
    TaskType.BINARY_CLASSIFICATION,
}, "RelArena currently supports only regression and binary classification tasks."
