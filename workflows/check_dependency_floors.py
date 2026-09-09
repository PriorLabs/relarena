"""Check core dependency wheel floors and supported backend constructors.

Build the workspace wheels first, then run with --wheels PATH --output PATH.
Installs dependencies into three clean environments without making API requests.
Core uses the lowest direct versions with compatible wheels for the interpreter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import tomllib


def main() -> None:
    """Install lowest-direct core and both backend floors, then run smoke checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    core = tomllib.loads((root / "packages/relarena-core/pyproject.toml").read_text())
    core_requirements = [
        requirement
        for requirement in core["project"]["dependencies"]
        if any(operator in requirement for operator in ("<", ">", "="))
    ]
    model_version = tomllib.loads(
        (root / "packages/tabpfn-rel/pyproject.toml").read_text()
    )["project"]["version"]
    cases = (
        ("core", [f"relarena-core=={core['project']['version']}", *core_requirements]),
        ("local", [f"tabpfn-rel[local]=={model_version}", "tabpfn==8.0.0"]),
        ("api", [f"tabpfn-rel[api]=={model_version}", "tabpfn-client==0.3.2"]),
    )
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    for name, requirements in cases:
        directory = output / name
        python = str(directory / "bin/python")
        commands = [
            ["uv", "venv", "--seed", "--python", sys.executable, str(directory)],
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--find-links",
                str(args.wheels.resolve()),
                *(
                    ["--resolution", "lowest-direct", "--only-binary", ":all:"]
                    if name == "core"
                    else []
                ),
                *requirements,
            ],
            [python, "-m", "pip", "check"],
        ]
        if name == "core":
            commands.extend(
                [
                    ["uv", "pip", "install", "--python", python, "pytest>=9.1.1"],
                    [
                        python,
                        "-m",
                        "pytest",
                        "-q",
                        "--import-mode=importlib",
                        str(
                            root
                            / "packages/relarena-core/tests/test_package_boundary.py"
                        ),
                        str(
                            root
                            / "packages/relarena-core/tests/test_standalone_runtime.py"
                        ),
                    ],
                ]
            )
        else:
            commands.append(
                [
                    python,
                    str(root / "packages/tabpfn-rel/workflows/check_backends.py"),
                    name,
                ]
            )
        with (output / f"{name}.log").open("w") as log:
            for command in commands:
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                subprocess.run(
                    command, cwd=output, env=env, stdout=log, stderr=log, check=True
                )
        print(f"{name}: passed", flush=True)


if __name__ == "__main__":
    main()
