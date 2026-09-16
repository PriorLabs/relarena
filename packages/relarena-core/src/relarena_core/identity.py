"""Source identity passed explicitly to preprocessing-owned cache-key factories.

`RunIdentity` describes the data and execution phase from which preprocessing
artifacts are produced. Entrypoints construct it, the runner passes it through
model construction, and each preprocessing module decides which fields belong
in its own artifact key.

A run identity is metadata, not a complete cache key and not a central key
policy. It intentionally contains more information than every artifact needs.
For example, task-dependent DFS and RelGT artifacts use task identity, while a
RelGNN graph may ignore it because the graph depends only on the censored
database. Preprocessors remain responsible for selecting actual dependencies
and adding their own algorithm versions.

The readable dataset and task names provide namespaces. Their fingerprints
distinguish different underlying data or task definitions without placing
absolute paths or expensive row hashes in keys. `phase` distinguishes
protocol views such as `inner`, `outer`, and `predict` when censoring or
inputs differ. `data_version` is an optional caller-supplied discriminator
for data changes that cheap fingerprints cannot observe.

RelBench identities use checked-in dataset and task checksums. Predictive-query
identities use a database-schema fingerprint and a task-specification
fingerprint; because the schema fingerprint deliberately ignores row contents,
callers using persistent caches should provide `data_version` when those
contents may change without a schema change.

Each preprocessing owner decides whether persistent use requires complete
fingerprints or whether it can derive a safe fallback from its actual inputs.
Unconfigured direct callers may omit identity and compute in private scratch.
Cache directories, miss policies, model names, serialization formats, and
preprocessing algorithm versions do not belong in this object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from relbench.base import Database


@dataclass(frozen=True)
class RunIdentity:
    """Readable source metadata that a preprocessor may use or ignore."""

    dataset: str
    dataset_fingerprint: str | None
    task: str | None
    task_fingerprint: str | None
    data_version: str | None = None
    phase: str | None = None

    def for_phase(self, phase: str | None) -> RunIdentity:
        """Return the same identity scoped to one execution phase."""
        return replace(self, phase=phase)


def metadata_fingerprint(value: Any) -> str:
    """Hash JSON metadata as sorted compact JSON with a 64-bit BLAKE2s digest."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2s(encoded, digest_size=8).hexdigest()


def database_schema_fingerprint(db: Database) -> str:
    """Fingerprint a user database's relational schema without hashing its rows."""
    schema = {
        name: {
            "columns": [
                (str(column), str(table.df[column].dtype)) for column in table.df
            ],
            "fkeys": sorted(table.fkey_col_to_pkey_table.items()),
            "pkey": table.pkey_col,
            "time": table.time_col,
        }
        for name, table in sorted(db.table_dict.items())
    }
    return metadata_fingerprint(schema)


def task_spec_fingerprint(task: Any) -> str:
    """Fingerprint training semantics of a user predictive-task specification."""
    fields = {
        name: str(getattr(task, name))
        for name in (
            "entity_table",
            "entity_col",
            "time_col",
            "target_col",
            "task_type",
            "timedelta",
            "query",
            "val_timestamp",
            "test_timestamp",
            "num_eval_timestamps",
        )
    }
    return metadata_fingerprint(fields)
