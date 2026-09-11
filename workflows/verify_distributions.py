"""Verify that built distributions contain RelArena's required package data."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def main() -> None:
    """Check required package data in the selected distribution directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheel = next(args.dist_dir.glob("relarena-*.whl"))
    sdist = next(args.dist_dir.glob("relarena-*.tar.gz"))
    source_root = Path("packages/relarena/src")
    spec_root = source_root / "relarena/userdb/relbench_v1"
    required = {
        "relarena/models/VENDORED-LICENSES",
        "relarena/checksums/relbench_v1_checksums.json",
        *(str(path.relative_to(source_root)) for path in spec_root.glob("*/*.yaml")),
    }

    core_required = {
        f"relarena_core/userdb/{name}.schema.json" for name in ("database", "task")
    }
    core_wheel = next(args.dist_dir.glob("relarena_core-*.whl"))
    core_sdist = next(args.dist_dir.glob("relarena_core-*.tar.gz"))
    with zipfile.ZipFile(core_wheel) as archive:
        _assert_present(core_wheel, archive.namelist(), core_required)
    with tarfile.open(core_sdist) as archive:
        _assert_present(core_sdist, archive.getnames(), core_required)

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
