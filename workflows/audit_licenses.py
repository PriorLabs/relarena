r"""Audit the licenses of every package in `uv.lock` against the permissive allowlist.

Reads declared license metadata from PyPI for each locked version, splits the lock
into core / extra / dev, and either gates on the result or writes a markdown report.
`--check` is the gate: it fails when a package that reaches users is neither
allowlisted-permissive nor carries an explicit `ACCEPTED` entry.

    python workflows/audit_licenses.py --check              # gate; non-zero on a finding
    python workflows/audit_licenses.py --output report.md   # write the report

The report is deliberately not committed: its `ACCEPTED` notes are legal positions
rather than facts, so it is generated per run (as a CI artifact) and `--output` has
no default.

Needs network access (queries pypi.org). Declared metadata only — this checks which
license a package declares, not whether we comply with it, and a package whose
verdict is REVIEW needs a human to read the actual license file upstream.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tomllib

LOCK_PATH = Path(__file__).resolve().parent.parent / "uv.lock"
ROOT_PACKAGE = "relarena"

#: Per-request retry budget for the PyPI metadata lookups. Kept short so a genuine
#: outage fails the audit in minutes rather than stalling a CI job: worst case is
#: FETCH_ATTEMPTS * FETCH_TIMEOUT_SECONDS per package, WORKERS at a time.
FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_SECONDS = 15
FETCH_BACKOFF_SECONDS = 2
WORKERS = 16

#: Patterns identifying a permissive license in a declared SPDX expression or a
#: `License ::` trove classifier. Word-anchored: a bare substring test would read
#: "MIT" out of "Limited" and pass a proprietary license as permissive.
PERMISSIVE_PATTERNS = (
    r"\bmit\b",
    r"\bmit-cmu\b",
    r"\bbsd\b",
    r"\b0bsd\b",
    r"\bapache\b",
    r"\bisc\b",
    r"python software foundation",
    r"\bpsf\b",
    r"\bzlib\b",
    r"\bunlicense\b",
    r"\bcc0\b",
    r"public domain",
    r"\bhpnd\b",
    r"historical permission notice",
)

#: Copyleft license *families*, matched against what a package declares itself to be.
#: Deliberately not matched against a pasted license body: bodies routinely enumerate
#: other licenses without being under them — pandas ships the PSF "GPL-compatible"
#: history table, scipy a bundled-licenses appendix listing libgfortran under GPL-3.0 —
#: so scanning bodies for a license name reports the dependencies of a dependency.
#: Checked before `PERMISSIVE_PATTERNS`, so a dual declaration like "MPL-2.0 AND MIT"
#: lands in BLOCKED and needs an explicit human decision.
BLOCKED_IDENTITY_PATTERNS = (
    r"\b[al]?gpl",
    r"gnu (general|lesser|affero)",
    r"\bmpl\b",
    r"mozilla public license",
    r"\bepl\b",
    r"eclipse public license",
    r"\bcddl\b",
    r"common development and distribution",
    r"\bsspl\b",
    r"server side public license",
    r"\bosl-?\d",
    r"open software license",
    r"\beupl\b",
)

#: Restriction *clauses*, matched against the full license text as well as the
#: declaration. Unlike a license name, one of these appearing anywhere in the text a
#: package ships is a restriction it is imposing, not a reference to someone else's
#: license — which is what catches an otherwise-permissive license with a term added
#: further down, the shape the audit exists for.
BLOCKED_CLAUSE_PATTERNS = (
    r"share[- ]?alike",
    r"\bby-sa\b",
    r"non-?commercial",
    r"\bby-nc\b",
)

#: Packages that the pattern lists cannot settle, with the reason each is acceptable
#: in a distributed Apache-2.0 package. Every entry is a human decision, checked
#: before the patterns and reproduced verbatim in the report. Each reason has to stand
#: alone: rows are sorted, so a note deferring to another ("same reasoning as ...")
#: can be numbered before the one it points at.
ACCEPTED: dict[str, str] = {
    "tabpfn": (
        "Prior Labs License v1.2 — Apache-2.0 plus paragraph 10 (enhanced "
        "attribution). Not plain Apache-2.0; confined to the rdblearn extra so a "
        "core install does not acquire its attribution obligation"
    ),
    "certifi": (
        "MPL-2.0 — file-level weak copyleft; used unmodified, so no source "
        "obligation reaches relarena's own code"
    ),
    "orjson": (
        "MPL-2.0 AND (Apache-2.0 OR MIT) — file-level weak copyleft; used "
        "unmodified, so no source obligation reaches relarena's own code. Pulled "
        "only by the optional rt integration"
    ),
    "tqdm": (
        "MPL-2.0 AND MIT — file-level weak copyleft; used unmodified, so no source "
        "obligation reaches relarena's own code"
    ),
    # A package whose `license` field concatenates the licenses of everything it
    # bundles will match clause patterns on text belonging to other projects. Only
    # scipy does so among the current dependencies; expect this to recur as other
    # packages vendor components, and confirm the declaration before adding one.
    "scipy": (
        "BSD-3-Clause per its own declaration. Its PyPI license field also carries the "
        "licenses of bundled components (libgfortran under GPL-3.0), so a clause scan "
        "of that text matches wording from those licenses rather than any term scipy "
        "imposes"
    ),
    "numpy": (
        "BSD-3-Clause per its own declaration. Its PyPI license field also carries "
        "licenses and notices for bundled components, so the non-commercial wording "
        "matched by the clause scan belongs to bundled material rather than a term "
        "NumPy imposes"
    ),
}

#: Prefixes covering the CUDA binary wheels torch pulls on Linux, each under an
#: NVIDIA EULA rather than an OSI license. Redistributed by the wheel index, not by
#: relarena; platform-gated and absent from a CPU-only or macOS install.
#: Same self-contained-reason rule as `ACCEPTED`.
ACCEPTED_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "nvidia-",
        "NVIDIA CUDA EULA — proprietary binary wheel pulled transitively by torch; "
        "redistributable as-is and not linked into relarena's own code",
    ),
    (
        "cuda-",
        "NVIDIA CUDA EULA — proprietary binary wheel pulled transitively by torch; "
        "redistributable as-is and not linked into relarena's own code",
    ),
)

#: Licenses for packages whose published metadata declares none, read by hand at the
#: locked version from the artifact or the upstream repository.
MANUAL_LICENSES: dict[str, str] = {
    # autogluon/tabarena @ 221c38d: packages/bencheval/pyproject.toml declares
    # Apache-2.0, matching this repository's LICENSE (no per-package file).
    "bencheval": "Apache-2.0",
    # googleapis/python-crc32c ships Apache-2.0 in LICENSE.
    "google-crc32c": "Apache-2.0",
    # The 0.2.19 wheel ships plain Apache-2.0 (no paragraph 10) in
    # dist-info/licenses/LICENSE while declaring no license in its metadata.
    "tabpfn-common-utils": "Apache-2.0",
    # The relational-transformer v1.8.0 source and wheels ship MIT in LICENSE,
    # while the published wheel declares no license in its metadata.
    "relational-transformer": "MIT",
    # The KurveRSC 0.1.1 wheel ships an MIT LICENSE and declares
    # `License-Expression: MIT`; PyPI's JSON API does not expose that PEP 639 field.
    "kurversc": "MIT",
}


def load_lock() -> dict:
    """Parse `uv.lock` into its raw TOML mapping."""
    with LOCK_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _closure(by_name: dict[str, dict], roots: list[str], with_extras: bool) -> set[str]:
    """Dependency closure over `roots`, optionally following optional-dependencies."""
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        pkg = by_name.get(name)
        if pkg is None:
            continue
        edges = list(pkg.get("dependencies", []))
        if with_extras:
            for extra_deps in pkg.get("optional-dependencies", {}).values():
                edges.extend(extra_deps)
        stack.extend(dep["name"] for dep in edges)
    return seen


def reachability(packages: list[dict]) -> dict[str, str]:
    """Classify each locked package as core, extra, or dev.

    core: installed by a plain `pip install relarena`. extra: only via an extra, so a
    user opts into it. dev: build- and test-time only, never distributed — audited but
    not gated. A package needed by both core and an extra counts as core.
    """
    by_name = {p["name"]: p for p in packages}
    core = _closure(by_name, [ROOT_PACKAGE], with_extras=False)
    runtime = _closure(by_name, [ROOT_PACKAGE], with_extras=True)
    return {
        p["name"]: "core"
        if p["name"] in core
        else "extra"
        if p["name"] in runtime
        else "dev"
        for p in packages
    }


def source_kind(pkg: dict) -> str:
    """The lock's source category for a package: registry, git, editable, ..."""
    source = pkg.get("source", {})
    if "registry" in source:
        return "pypi"
    return next(iter(source), "unknown")


class MetadataUnavailable(Exception):
    """PyPI could not be reached, as opposed to declaring no license."""


def _fetch_pypi_info(name: str, version: str) -> dict | None:
    """One package's PyPI `info` mapping, retrying transient transport failures.

    Returns None when the version exists but has no metadata to read. Raises
    `MetadataUnavailable` once the retries are spent.

    Retrying per request rather than per run: an audit makes one call per locked
    package, so a single reset connection or CDN 5xx is far likelier than the index
    being down, and re-requesting the one that failed beats redoing all of them.
    """
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:
                return json.load(response)["info"]
        except urllib.error.HTTPError as exc:
            # A missing version is real metadata ("nothing declared here") and will
            # not change on a retry; a 5xx or a rate limit is the index failing us.
            if exc.code == 404:
                return None
            reason: object = f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            reason = exc.reason
        except (KeyError, json.JSONDecodeError):
            return None

        if attempt == FETCH_ATTEMPTS:
            raise MetadataUnavailable(f"{name} {version}: {reason}")
        # Exponential with equal jitter. Requests go out WORKERS at a time, so a
        # fixed delay would retry a whole batch in lockstep and hit a struggling
        # index with the same burst that just failed; the jitter spreads them out.
        ceiling = FETCH_BACKOFF_SECONDS * 2 ** (attempt - 1)
        time.sleep(random.uniform(ceiling / 2, ceiling))
    raise AssertionError("unreachable")


def fetch_pypi_license(name: str, version: str) -> tuple[str | None, str]:
    """Declared license for one locked version, as a (display, full text) pair.

    Prefers the PEP 639 `license-expression`, then the `License ::` classifiers, then
    the free-text `license` field. Display is a single line fit for a table cell; the
    full text is what gets pattern-matched, since a project that pastes an entire
    license may bury an added restriction hundreds of lines below a permissive-looking
    first line.
    """
    # A local version segment marks an alternate build of the same upstream release
    # (torch 2.13.0+cpu from the pytorch-cpu index), which PyPI does not serve. The
    # license belongs to the release, so ask about that.
    info = _fetch_pypi_info(name, version.partition("+")[0])
    if info is None:
        return None, ""

    expression = info.get("license_expression") or ""
    classifiers = [
        c.split("::")[-1].strip()
        for c in info.get("classifiers", [])
        if c.startswith("License ::")
    ]
    body = info.get("license") or ""

    # Everything the package says about its license, whichever field it used. The
    # display line follows the precedence order below, but the pattern scan gets all
    # of it: a project can declare an Apache classifier and paste a *modified* Apache
    # license into `license`, which is exactly the shape the audit needs to catch, so
    # picking one field for the scan would let the added restriction through.
    full_text = "\n".join(
        part for part in (expression, " / ".join(classifiers), body) if part
    )

    if expression:
        return expression, full_text
    if classifiers:
        return " / ".join(classifiers), full_text
    # Projects that paste the whole license text start it with a blank line, so take
    # the first line with content rather than line zero.
    for line in body.splitlines():
        if stripped := line.strip():
            return _abbreviate(stripped), full_text
    return None, ""


def _abbreviate(text: str, limit: int = 60) -> str:
    """Shorten a free-text license line to fit a table cell.

    Cuts on a word boundary and marks the elision, so a pasted license body reads as
    deliberately abbreviated rather than as a corrupted value.
    """
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;.")
    return f"{head} […]"


def classify(name: str, display: str | None, full_text: str) -> tuple[str, str]:
    """Map a declared license to a (verdict, note) pair.

    Verdicts: PERMISSIVE (allowlisted), ACCEPTED (explicit human decision),
    BLOCKED (copyleft or non-commercial), REVIEW (unrecognized or absent).

    The three pattern lists deliberately match against different things, and it is not
    a bug that they differ:

      * clause patterns scan the whole license text, because a restriction anywhere in
        the text a package ships is one it imposes — this is what catches a permissive
        license with a term added far below its first line;
      * identity patterns scan only the declaration, because bodies enumerate other
        licenses (a bundled-licenses appendix would otherwise report GPL for scipy);
      * permissive patterns scan only the declaration, so a pasted Apache-2.0 body
        cannot wave itself through on the word "apache" alone.

    Conservative in each direction, which is why some packages need an `ACCEPTED` entry
    rather than passing automatically.
    """
    if name in ACCEPTED:
        return "ACCEPTED", ACCEPTED[name]
    if display:
        for pattern in BLOCKED_CLAUSE_PATTERNS:
            if re.search(pattern, full_text or display, re.IGNORECASE):
                return "BLOCKED", f"matched /{pattern}/ in the license text"
        for pattern in BLOCKED_IDENTITY_PATTERNS:
            if re.search(pattern, display, re.IGNORECASE):
                return "BLOCKED", f"declared as /{pattern}/"
        for pattern in PERMISSIVE_PATTERNS:
            if re.search(pattern, display, re.IGNORECASE):
                return "PERMISSIVE", ""
    # Prefix rules are a fallback, not an override: a CUDA-family package that
    # declares a real permissive license is classified on that license instead.
    for prefix, reason in ACCEPTED_PREFIXES:
        if name.startswith(prefix):
            return "ACCEPTED", reason
    if not display:
        return "REVIEW", "no license declared in metadata"
    return "REVIEW", "license not on the allowlist"


def audit() -> list[dict]:
    """Resolve, classify, and sort every locked package."""
    packages = load_lock()["package"]
    tiers = reachability(packages)

    def resolve(pkg: dict) -> dict:
        name, version = pkg["name"], pkg["version"]
        kind = source_kind(pkg)
        if name == ROOT_PACKAGE:
            display, full_text = "Apache-2.0", "Apache-2.0"
        else:
            display, full_text = (
                fetch_pypi_license(name, version) if kind == "pypi" else (None, "")
            )
            display = display or MANUAL_LICENSES.get(name)
        verdict, note = classify(name, display, full_text)
        return {
            "name": name,
            "version": version,
            "source": kind,
            "license": display,
            "verdict": verdict,
            "note": note,
            "tier": tiers[name],
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(resolve, packages))

    tier_order = {"core": 0, "extra": 1, "dev": 2}
    order = {"BLOCKED": 0, "REVIEW": 1, "ACCEPTED": 2, "PERMISSIVE": 3}
    rows.sort(key=lambda r: (tier_order[r["tier"]], order[r["verdict"]], r["name"]))
    return rows


def findings(rows: list[dict]) -> list[dict]:
    """Distributed packages that are neither permissive nor explicitly accepted."""
    return [
        r
        for r in rows
        if r["tier"] in ("core", "extra") and r["verdict"] in ("BLOCKED", "REVIEW")
    ]


def _summarize(rows: list[dict]) -> str:
    """One-line verdict tally for a tier."""
    counts = {
        v: sum(1 for r in rows if r["verdict"] == v)
        for v in ("PERMISSIVE", "ACCEPTED", "REVIEW", "BLOCKED")
    }
    return (
        f"{len(rows)} packages — {counts['PERMISSIVE']} permissive, "
        f"{counts['ACCEPTED']} accepted, {counts['REVIEW']} to review, "
        f"{counts['BLOCKED']} blocked."
    )


def render(rows: list[dict]) -> str:
    """Render the audit as the markdown report."""
    tiers = {t: [r for r in rows if r["tier"] == t] for t in ("core", "extra", "dev")}
    notes: dict[str, int] = {}

    def table(tier_rows: list[dict]) -> list[str]:
        out = [
            "| Package | Version | Source | Declared license | Verdict | Note |",
            "|---|---|---|---|---|---|",
        ]
        for r in tier_rows:
            # One footnote per distinct reason: the CUDA rationale would otherwise
            # repeat across a dozen near-identical rows.
            marker = ""
            if r["note"]:
                marker = f"[^{notes.setdefault(r['note'], len(notes) + 1)}]"
            out.append(
                f"| {r['name']} | {r['version']} | {r['source']} | "
                f"{r['license'] or '—'} | {r['verdict']} | {marker} |"
            )
        return out

    lines = [
        "# Dependency licenses",
        "",
        "Declared licenses for every package in `uv.lock`, generated by",
        "`workflows/audit_licenses.py` on every run that touches the dependency set.",
        "Not committed, so it always describes the lock it was generated from. The",
        "`--check` mode gates on everything that reaches users (core + extras).",
        "",
        "Three tiers: **core** is what a plain install pulls, **extra** is what a user",
        "opts into via an extra, and **dev** is build- and test-time only and never",
        "distributed (audited, but it does not gate). Extras are followed at every",
        "level, so the extra tier is a deliberate over-approximation.",
        "",
        f"- core: {_summarize(tiers['core'])}",
        f"- extra: {_summarize(tiers['extra'])}",
        f"- dev: {_summarize(tiers['dev'])}",
        "",
        "## Core",
        "",
        "No core package carries a strong copyleft or a non-commercial clause. The",
        "entries that are not plain-permissive are all ACCEPTED with a reason below:",
        "MPL-2.0 (file-level, used unmodified) and the NVIDIA CUDA wheels that `torch`",
        "pulls on Linux. `tabpfn` — the one dependency under the Prior Labs License,",
        "whose paragraph 10 obliges downstream attribution — is deliberately *not*",
        "here; it sits in the `rdblearn` extra (see `docs/licensing.md`).",
        "",
        *table(tiers["core"]),
        "",
        "## Extras",
        "",
        *table(tiers["extra"]),
        "",
        "## Dev-only (not distributed)",
        "",
        *table(tiers["dev"]),
        "",
    ]
    for note, index in notes.items():
        lines.append(f"[^{index}]: {note}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Gate on the audit with `--check`, or write the report to `--output`.

    Exit codes: 0 clean, 1 license findings, 2 the audit could not be completed.
    """
    parser = argparse.ArgumentParser(prog="audit_licenses", description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero on a BLOCKED/REVIEW package that reaches users, instead "
        "of writing the report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="path to write the markdown report to. No default: the report is a "
        "generated artifact and is deliberately not committed",
    )
    args = parser.parse_args(argv)

    if not args.check and args.output is None:
        parser.error("one of --check or --output is required")

    try:
        rows = audit()
    except MetadataUnavailable as exc:
        # Distinct from a finding: we learned nothing, rather than learning something
        # bad. Never let an index outage read as a license problem.
        print(f"could not complete the audit: {exc}", file=sys.stderr)
        return 2

    hits = findings(rows)
    for r in hits:
        print(
            f"{r['verdict']}: {r['name']} {r['version']} — "
            f"{r['license'] or 'no license declared'} ({r['note']})"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render(rows))
        print(f"wrote {args.output} ({len(rows)} packages)")

    distributed = sum(r["tier"] in ("core", "extra") for r in rows)
    print(f"{len(hits)} finding(s) among {distributed} distributed packages")
    return 1 if hits and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
