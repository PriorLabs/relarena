# Releasing RelArena to PyPI

Releases are cut by [`.github/workflows/release.yml`](../.github/workflows/release.yml).
Pushing a `v<version>` tag builds the distributions, verifies them, publishes to
PyPI, and cuts a GitHub release with the same artifacts attached. Nothing is
published from a laptop, and no PyPI API token is stored in the repository —
uploads authenticate with [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so only this
workflow, in this repository, can publish `relarena`.

## One-time setup

Someone with owner rights on the PyPI project does this once; it does not need
repeating per release.

1. **PyPI** → the `relarena` project → *Publishing* → add a GitHub publisher:

   | Field | Value |
   | --- | --- |
   | Owner | `PriorLabs` |
   | Repository | `relarena` |
   | Workflow | `release.yml` |
   | Environment | `pypi` |

2. **TestPyPI** → same form at <https://test.pypi.org/manage/account/publishing/>,
   with environment `testpypi`. (TestPyPI is a separate account and a separate
   project registration.)

3. **GitHub** → *Settings* → *Environments* → create `pypi` and `testpypi`. Add
   required reviewers on `pypi` if a release should need a second pair of eyes;
   the workflow then pauses before the upload step until someone approves.

The environment names must match on both sides — a mismatch fails at the upload
step with `invalid-publisher`, after the build has already succeeded.

## Cutting a release

1. Bump the version in the two places that carry it, plus the lock file:

   ```bash
   # pyproject.toml   -> [project] version
   # src/relarena/__init__.py -> __version__
   uv lock            # refreshes the relarena entry in uv.lock
   ```

   The workflow refuses to build if those two versions disagree, or if the tag
   does not equal `v<version>`. PyPI never allows a version to be re-uploaded,
   so this is checked before anything is built rather than after.

   Version strings follow [PEP 440](https://peps.python.org/pep-0440/): `0.1.0`,
   `0.1.0a1`, `0.1.0rc1`. Anything ending in `a`/`b`/`rc`/`.dev` is marked as a
   prerelease on the GitHub release and is not installed by a bare
   `pip install relarena`.

2. Merge the bump to `main` and let CI pass.

3. Tag and push:

   ```bash
   git tag v0.1.0a3 -m "relarena 0.1.0a3"
   git push origin v0.1.0a3
   ```

4. Watch the run. On success the package is live at
   <https://pypi.org/project/relarena/> and a GitHub release is drafted from the
   commits since the previous tag.

## Dry runs

`workflow_dispatch` (the *Run workflow* button) takes a `target` input:

- `none` — build and run every verification, publish nothing. Use this to check
  a release candidate before tagging.
- `testpypi` — same, then upload to TestPyPI. Useful for rehearsing the install
  path end to end:

  ```bash
  pip install --index-url https://test.pypi.org/simple/ \
              --extra-index-url https://pypi.org/simple/ relarena
  ```

  The extra index is required because TestPyPI does not mirror `relarena`'s
  dependencies.
- `pypi` — publish the current branch's version to PyPI without a tag. Prefer
  tagging; this exists for recovery when a tag push failed after the tag was
  already consumed.

## What the release gates on

Beyond the version check, the build job runs three verifications before anything
can be published:

- `workflows/verify_distributions.py` — the wheel and sdist both carry the
  package data the harness needs at runtime (vendored licenses, dataset
  checksums, userdb JSON schemas and task YAML).
- `twine check --strict` — the rendered README and metadata are valid, so the
  PyPI project page does not land broken.
- A smoke install of the built wheel into a clean environment, which imports
  every name in `relarena.__all__` and runs `relarena --help`. This is the check
  that catches a release which builds fine but ships an API older than the docs
  describe — the failure mode behind the `0.0.1a1` alpha, where the cookbook
  called `PredictiveQuery.compute_test_labels` and the published package did not
  have it.

## Known limitation: the `leaderboard` and `plots` extras

Both extras depend on `bencheval`, which is **not published on PyPI** — this
repository resolves it from the TabArena git repository via
`[tool.uv.sources]`. That source is a local resolution hint and is not recorded
in the published metadata, so

```bash
pip install "relarena[leaderboard]"   # fails: No matching distribution for bencheval
```

fails for anyone installing from PyPI. Leaderboard aggregation and plots
currently work only from a git checkout with `uv sync`, which the README's
Installation section states. Nothing in the release workflow can fix this —
it needs `bencheval` on PyPI, or the extras vendored differently. Keep the
README caveat in step with reality until then.
