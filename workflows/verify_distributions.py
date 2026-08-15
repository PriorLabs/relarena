"""Verify that built distributions contain RelArena's required package data."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def main() -> None:
    """Check the wheel and sdist under ``dist/``."""
    wheel = next(Path("dist").glob("*.whl"))
    sdist = next(Path("dist").glob("*.tar.gz"))
    spec_root = Path("src/relarena/userdb/relbench_v1")
    required = {
        "relarena/models/VENDORED-LICENSES",
        "relarena/checksums/relbench_v1_checksums.json",
        "relarena/userdb/database.schema.json",
        "relarena/userdb/task.schema.json",
        *(str(path.relative_to("src")) for path in spec_root.glob("*/*.yaml")),
    }

    with zipfile.ZipFile(wheel) as archive:
        _assert_present(wheel, archive.namelist(), required)
    with tarfile.open(sdist) as archive:
        _assert_present(sdist, archive.getnames(), required)


def _assert_present(artifact: Path, entries: list[str], required: set[str]) -> None:
    """Raise when an artifact omits any required package-data path."""
    missing = sorted(
        required_path
        for required_path in required
        if not any(entry.endswith(required_path) for entry in entries)
    )
    if missing:
        raise RuntimeError(f"{artifact} is missing required files: {missing}")


if __name__ == "__main__":
    main()
