# Contributing to RelArena

RelArena is an alpha-stage open-source benchmark. Bug reports, documentation
improvements, new model integrations, and reproducibility fixes are welcome.

## Development setup

RelArena requires Python 3.11 and uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev --group cpu --extra leaderboard --extra plots
uv run pre-commit install
```

On macOS, prefix test and CLI commands with `OMP_NUM_THREADS=1` to avoid the
known conflict between the OpenMP runtimes bundled by PyTorch and LightGBM.

Before opening a pull request, run:

```bash
uv run ruff format --check .
uv run ruff check .
OMP_NUM_THREADS=1 uv run pytest
uv build
```

Releases to PyPI are cut by pushing a `v<version>` tag; the procedure and the
one-time publisher setup are in [docs/releasing.md](docs/releasing.md).

For model integrations, follow [docs/adding-a-model.md](docs/adding-a-model.md).
Keep optional dependencies lazy, document provenance for adapted or vendored
code, and include focused tests. By contributing, you agree that your changes
are licensed under this repository's Apache-2.0 license.
