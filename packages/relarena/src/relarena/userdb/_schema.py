"""Load and validate userdb YAML documents against their JSON Schemas."""

from __future__ import annotations

import json
from importlib.resources import files

import jsonschema


def load_schema(name: str) -> dict:
    """Load a JSON Schema shipped alongside this package, by file name."""
    return json.loads((files("relarena.userdb") / name).read_text())


def validate(raw: object, schema: dict, *, kind: str) -> None:
    """Validate a loaded YAML document against `schema`, or raise a clear `ValueError`.

    `kind` names the document in the error (e.g. `task`, `database`). yaml.safe_load
    turns unquoted timestamps into date/datetime objects; normalizing to JSON types
    (those become ISO strings) lets the schema's string-typed fields accept them,
    matching what `pd.Timestamp` accepts downstream.
    """
    normalized = json.loads(json.dumps(raw, default=str))
    try:
        jsonschema.validate(normalized, schema)
    except jsonschema.ValidationError as e:
        loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(f"Invalid {kind} YAML at {loc}: {e.message}") from e
