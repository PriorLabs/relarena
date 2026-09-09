"""Recorded benchmark identities and shared source metadata."""

import json
from functools import cache
from pathlib import Path

from relarena_core.identity import (
    RunIdentity,
    metadata_fingerprint,
)

CHECKSUMS_PATH = Path(__file__).with_name("checksums") / "relbench_v1_checksums.json"


@cache
def _recorded_checksums(path: Path = CHECKSUMS_PATH) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text())


def relbench_run_identity(dataset: str, task: str) -> RunIdentity:
    """Build a cheap native identity from the checked-in RelBench checksums."""
    record = _recorded_checksums().get(f"{dataset}/{task}")
    if record is None:
        return RunIdentity(dataset, None, task, None)
    dataset_fingerprint = f"{record['inner_db']:016x}-{record['outer_db']:016x}"
    task_values = {
        key: value for key, value in record.items() if not key.endswith("_db")
    }
    return RunIdentity(
        dataset, dataset_fingerprint, task, metadata_fingerprint(task_values)
    )


__all__ = [
    "RunIdentity",
    "relbench_run_identity",
]
