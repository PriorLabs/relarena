"""Check built packages in three isolated installations on every CI run.

Installs core, the benchmark and TabPFN-Rel separately to catch missing runtime
dependencies, package data and entry points hidden by a workspace install. Checks
wheel metadata, schemas, notices, plugin discovery and the benchmark CLI's task
listing. Backend extras and real inference are not exercised.

Run from the workspace with ``python workflows/check_package_split.py --output PATH``.
Downloads dependencies, but does not download model weights or call an API.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

CASES = (
    ("core", "relarena-core"),
    ("base", "relarena"),
    ("model", "tabpfn-rel"),
)


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    """Run a checked command and retain its output for review."""
    with log.open("a") as output:
        output.write("\n$ " + " ".join(command) + "\n")
        output.flush()
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def check_artifacts(wheels: Path) -> list[str]:
    """Verify dependency direction, package ownership, schemas and notices."""
    pins = []
    for name in ("relarena", "relarena_core", "tabpfn_rel"):
        wheel = next(wheels.glob(f"{name}-*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata = BytesParser().parsebytes(
                archive.read(next(n for n in names if n.endswith("/METADATA")))
            )
            pins.append(f"{metadata['Name']}=={metadata['Version']}")
            requirements = metadata.get_all("Requires-Dist", [])
            assert not any(" @ " in req for req in requirements), requirements
            for namespace in ("relarena", "relarena_core", "tabpfn_rel"):
                if namespace != name:
                    assert not any(n.startswith(namespace + "/") for n in names)
            for notice in ("LICENSE", "NOTICE"):
                assert any(n.endswith("/licenses/" + notice) for n in names)
            if name == "relarena":
                plugin_requirements = [
                    r for r in requirements if r.startswith("tabpfn-rel")
                ]
                assert len(plugin_requirements) == 2, plugin_requirements
                assert all("extra ==" in r for r in plugin_requirements)
                assert any(r.startswith("relarena-core") for r in requirements)
                assert "relarena/checksums/relbench_v1_checksums.json" in names
                assert "relarena/models/VENDORED-LICENSES" in names
                assert any(n.endswith("/db.yaml") for n in names)
            else:
                assert not any(
                    r.startswith(("relarena>", "relarena=", "relarena[", "relarena "))
                    for r in requirements
                )
                if name == "relarena_core":
                    assert not any(r.startswith("tabpfn") for r in requirements)
                    for schema in ("database", "task"):
                        assert f"relarena_core/userdb/{schema}.schema.json" in names
                else:
                    assert any(r.startswith("relarena-core") for r in requirements)
            if name != "relarena_core":
                entrypoints = archive.read(
                    next(n for n in names if n.endswith("/entry_points.txt"))
                ).decode()
                assert "[relarena.models]" in entrypoints
                target = "relarena.models" if name == "relarena" else "tabpfn_rel.model"
                assert target in entrypoints
        with tarfile.open(next(wheels.glob(f"{name}-*.tar.gz"))) as archive:
            names = archive.getnames()
            assert any(n.endswith("/pyproject.toml") for n in names)
            assert any(n.endswith(f"/src/{name}/__init__.py") for n in names)
    return pins


def main() -> None:
    """Build distributions and check each package in an isolated environment."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "wheels"
    wheels.mkdir()
    log = output / "commands.log"
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(output / "uv-cache")),
    }
    env.pop("PYTHONPATH", None)
    for source in (
        root / "packages/relarena-core",
        root / "packages/relarena",
        root / "packages/tabpfn-rel",
    ):
        run(["uv", "build", "--out-dir", str(wheels)], source, env, log)
    pins = check_artifacts(wheels)
    constraints = output / "constraints.txt"
    constraints.write_text("\n".join(pins) + "\n")
    results = []
    for name, requirement in CASES:
        directory = output / name
        print(f"Testing {name}", flush=True)
        run(
            ["uv", "venv", "--seed", "--python", args.python, str(directory)],
            output,
            env,
            log,
        )
        python = str(directory / "bin/python")
        install = ["uv", "pip", "install", "--python", python]
        run(
            [
                *install,
                "--find-links",
                str(wheels),
                "--constraint",
                str(constraints),
                requirement,
            ],
            output,
            env,
            log,
        )
        run([python, "-m", "pip", "check"], output, env, log)
        host = name == "base"
        has_model = name == "model"
        code = f"""
import importlib.metadata as metadata
import importlib.util
import sys
import relarena_core
from relarena_core.userdb import PredictiveQuery
from relarena_core.userdb._schema import load_schema
installed = {{d.metadata['Name'].lower().replace('_', '-') for d in metadata.distributions()}}
assert 'relarena-core' in installed
assert ('relarena' in installed) == {host!r}
assert ('tabpfn-rel' in installed) == {has_model!r}
assert not relarena_core.registry.names()
assert load_schema('task.schema.json')['type'] == 'object'
assert load_schema('database.schema.json')['type'] == 'object'
if not {host!r}:
    assert importlib.util.find_spec('relarena') is None
if {host!r}:
    import relarena
    from relarena.userdb import PredictiveQuery as HostQuery
    assert HostQuery is PredictiveQuery
    assert relarena.RelArenaModel is relarena_core.RelArenaModel
    assert relarena.registry is relarena_core.registry
if {has_model!r}:
    import tabpfn_rel
    assert tabpfn_rel.PredictiveQuery is PredictiveQuery
relarena_core.discover_models()
relarena_core.discover_models()
assert ('tabpfn-rel-local' in relarena_core.registry) == {has_model!r}
assert ('rdblearn' in relarena_core.registry) == {host!r}
assert 'tabpfn' not in sys.modules
assert 'tabpfn_client' not in sys.modules
if {has_model!r}:
    assert relarena_core.registry.get('tabpfn-rel-local') is tabpfn_rel.TabPFNRelLocalModel
print('Verified', sys.executable, sorted(installed))
"""
        run([python, "-c", code], output, env, log)
        if host:
            run(
                [python, "-m", "relarena.cli", "--list", "--datasets", "rel-f1"],
                output,
                env,
                log,
            )
        results.append({"case": name, "passed": True})
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"All {len(results)} installation paths passed. Results: {output / 'results.json'}"
    )


if __name__ == "__main__":
    main()
