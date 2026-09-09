"""Recorded benchmark source identities."""

from relarena.identity import relbench_run_identity


def test__relbench_run_identity__recorded_task__is_complete_and_stable() -> None:
    first = relbench_run_identity("rel-f1", "driver-dnf")
    second = relbench_run_identity("rel-f1", "driver-dnf")
    assert first == second
    assert first.dataset_fingerprint is not None
    assert len(first.dataset_fingerprint.split("-")) == 2
    assert first.task_fingerprint is not None
    assert len(first.task_fingerprint) == 16
