"""Tests for the dependency-license gate (`workflows/audit_licenses.py`).

The cases that matter are the ones where a wrong answer is invisible: a package
declaring a permissive license while shipping a restricted one, and the mirror
image where a license body merely cites a copyleft license belonging to something
it bundles. Both are exercised against a stubbed PyPI response.

The script is not importable as part of the package, so it is loaded from its path.
"""

from __future__ import annotations

import importlib.util
import io
import json
import types
import urllib.error
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "workflows" / "audit_licenses.py"


@pytest.fixture(scope="module")
def audit() -> types.ModuleType:
    """The audit script, loaded from its path rather than imported."""
    spec = importlib.util.spec_from_file_location("audit_licenses", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_pypi(
    audit: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    info: dict[str, Any],
    failures: list[Exception] | None = None,
) -> list[str]:
    """Point the module's urlopen at `info`, optionally failing first.

    Returns the list of requested URLs so a test can assert how many attempts ran.
    """
    calls: list[str] = []
    pending = list(failures or [])

    def fake_urlopen(url: str, timeout: float | None = None) -> Any:
        calls.append(url)
        if pending:
            raise pending.pop(0)

        class _Response:
            def __enter__(self) -> io.StringIO:
                return io.StringIO(json.dumps({"info": info}))

            def __exit__(self, *exc: object) -> bool:
                return False

        return _Response()

    monkeypatch.setattr(audit.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(audit, "FETCH_BACKOFF_SECONDS", 0)
    return calls


def test__classify__permissive_classifier_over_restricted_body__is_blocked(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failure mode the gate exists to prevent: the trove classifier says MIT, and
    # the restriction is thousands of characters into the text the package ships.
    # Reading only the classifier would report this as permissive.
    _stub_pypi(
        audit,
        monkeypatch,
        {
            "classifiers": ["License :: OSI Approved :: MIT License"],
            "license": "MIT License\n"
            + "filler\n" * 400
            + "10. Additional term: non-commercial use only.\n",
        },
    )
    display, full_text = audit.fetch_pypi_license("sneaky", "1.0")

    assert display == "MIT License"
    assert audit.classify("sneaky", display, full_text)[0] == "BLOCKED"


def test__classify__permissive_classifier_and_benign_body__is_permissive(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pypi(
        audit,
        monkeypatch,
        {
            "classifiers": ["License :: OSI Approved :: MIT License"],
            "license": "MIT License\n\nPermission is hereby granted, free of charge.",
        },
    )
    display, full_text = audit.fetch_pypi_license("ordinary", "1.0")

    assert audit.classify("ordinary", display, full_text)[0] == "PERMISSIVE"


def test__classify__body_citing_a_copyleft_dependency__is_not_blocked(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The mirror image, and why license *names* are matched against the declaration
    # only. Several scientific packages append the licenses of everything they bundle,
    # so a GPL notice in the body describes vendored code rather than the package's own
    # terms. Matching names in the body would block pandas and scipy.
    _stub_pypi(
        audit,
        monkeypatch,
        {
            "classifiers": ["License :: OSI Approved :: BSD License"],
            "license": (
                "BSD 3-Clause License\n\nCopyright (c) 2019\n\n"
                "This distribution bundles the following third-party components:\n"
                "  Name: libgfortran\n"
                "  License: GPL-3.0-or-later WITH GCC-exception-3.1\n"
                "  GNU General Public License for more details.\n"
            ),
        },
    )
    display, full_text = audit.fetch_pypi_license("sciencey", "1.0")

    assert audit.classify("sciencey", display, full_text)[0] == "PERMISSIVE"


def test__classify__declared_as_copyleft__is_blocked(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Spellings that turn up in free-text license fields, where a trailing word
    # boundary after "gpl" would not match.
    for declared in ("AGPLv3", "GPLv2", "LGPL-2.1", "GPL-3.0-or-later"):
        assert audit.classify("x", declared, declared)[0] == "BLOCKED", declared


def test__classify__accepted_bundled_license_notice__is_accepted(
    audit: types.ModuleType,
) -> None:
    verdict, note = audit.classify(
        "numpy",
        "BSD License",
        "BSD 3-Clause License\nBundled component: non-commercial use only",
    )

    assert verdict == "ACCEPTED"
    assert "bundled" in note.lower()


def test__manual_licenses__covers_kurversc_pep639_metadata_gap(
    audit: types.ModuleType,
) -> None:
    assert audit.MANUAL_LICENSES["kurversc"] == "MIT"


def test__fetch_pypi_license__transient_failure__retries_and_recovers(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_pypi(
        audit,
        monkeypatch,
        {"license_expression": "MIT"},
        failures=[urllib.error.URLError("connection reset")],
    )
    assert audit.fetch_pypi_license("flaky", "1.0") == ("MIT", "MIT")
    assert len(calls) == 2


def test__fetch_pypi_license__missing_version__does_not_retry(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 404 is metadata ("this version declares nothing"), not a transport failure, so
    # retrying it would cost a request and a backoff per absent package.
    calls = _stub_pypi(
        audit,
        monkeypatch,
        {},
        failures=[urllib.error.HTTPError("u", 404, "not found", {}, None)],  # type: ignore[arg-type]
    )
    assert audit.fetch_pypi_license("absent", "1.0") == (None, "")
    assert len(calls) == 1


def test__fetch_pypi_license__local_version_segment__queries_the_release(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An extra index publishes alternate builds (torch's +cpu wheels) under a local
    # version PyPI has never seen, so querying it verbatim reads as "declares nothing".
    calls = _stub_pypi(audit, monkeypatch, {"license_expression": "BSD-3-Clause"})
    assert audit.fetch_pypi_license("torch", "2.13.0+cpu") == (
        "BSD-3-Clause",
        "BSD-3-Clause",
    )
    assert calls == ["https://pypi.org/pypi/torch/2.13.0/json"]


def test__fetch_pypi_license__index_unreachable__raises_rather_than_reporting(
    audit: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreachable index must not read as a license finding.
    calls = _stub_pypi(
        audit,
        monkeypatch,
        {},
        failures=[urllib.error.URLError("down")] * audit.FETCH_ATTEMPTS,
    )
    with pytest.raises(audit.MetadataUnavailable):
        audit.fetch_pypi_license("unreachable", "1.0")
    assert len(calls) == audit.FETCH_ATTEMPTS
