# Contributing to RelArena

RelArena is an alpha-stage open-source benchmark. Bug reports, documentation
improvements, new model integrations, and reproducibility fixes are welcome.

## Development setup

RelArena supports Python 3.11 and 3.12 and uses [uv](https://docs.astral.sh/uv/).
The workspace contains three sibling distributions under `packages/`: `relarena`,
`relarena-core`, and `tabpfn-rel`. Each has its own source tree and tests; the root
owns shared development tooling and the lockfile. Install all workspace members
for the combined test suite. To install only one member with pip, use its directory,
for example `pip install ./packages/relarena` after its dependency wheels are available.

```bash
uv sync --all-packages --group dev --group cpu --extra leaderboard --extra plots
uv run --all-packages pre-commit install
```

On macOS, prefix test and CLI commands with `OMP_NUM_THREADS=1` to avoid the
known conflict between the OpenMP runtimes bundled by PyTorch and LightGBM.

Before opening a pull request, run:

```bash
uv run --all-packages ruff format --check .
uv run --all-packages ruff check .
OMP_NUM_THREADS=1 uv run --all-packages pytest
uv build --all-packages
```

For model integrations, follow [docs/adding-a-model.md](docs/adding-a-model.md).
Open the pull request from a fork owned by your personal account with **Allow
edits from maintainers** enabled, so maintainers can push rerun results to your
branch (see [Submitting](docs/adding-a-model.md#10-submitting)).
Keep optional dependencies lazy, document provenance for adapted or vendored
code, and include focused tests. By contributing, you agree that your changes
are licensed under this repository's Apache-2.0 license.
