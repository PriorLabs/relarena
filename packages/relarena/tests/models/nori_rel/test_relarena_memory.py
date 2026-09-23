"""The NoriDFS inference memory policy and its post-prediction audit."""

from __future__ import annotations

from typing import Any

import pytest

from relarena.models.nori_rel.memory import (
    NORIDFS_MEMORY_POLICY,
    validate_memory_report,
)
from relarena.models.nori_rel.model import GPU_BUDGET_GB_ENV, _memory_policy


def test_noridfs_policy_stays_independent_of_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GPU_BUDGET_GB_ENV, "110")
    baseline = _memory_policy()

    assert baseline["context_row_chunk"] is None
    assert baseline["gpu_budget_absolute_gb"] == 110
    assert NORIDFS_MEMORY_POLICY["context_row_chunk"] == 512
    assert NORIDFS_MEMORY_POLICY["gpu_budget_absolute_gb"] == 0
    assert NORIDFS_MEMORY_POLICY["host_budget_absolute_gb"] == 64
    assert NORIDFS_MEMORY_POLICY["cache_dtype"] == "bf16"


@pytest.mark.parametrize("rung", ["no_cache", "offload_bf16", "plain_loop"])
def test_exact_report_accepts_bit_exact_rungs(rung: str) -> None:
    report = {
        **NORIDFS_MEMORY_POLICY,
        **(
            {
                "gpu_budget_absolute_gb": None,
                "host_budget_absolute_gb": None,
                "offload_to_host": False,
            }
            if rung != "offload_bf16"
            else {}
        ),
        "rung": rung,
        "dropped_context_rows": 0,
    }

    audit = validate_memory_report(report, dict(NORIDFS_MEMORY_POLICY))

    assert audit["memory_rung_actual"] == rung


@pytest.mark.parametrize(
    "updates",
    [
        {"rung": "resident_bf16"},
        {"rung": "resident_int8", "cache_dtype": "int8"},
        {"rung": "offload_int8", "cache_dtype": "int8"},
        {"rung": "offload_bf16", "dropped_context_rows": 1},
    ],
)
def test_exact_report_rejects_fallbacks(updates: dict[str, Any]) -> None:
    report = {
        **NORIDFS_MEMORY_POLICY,
        "rung": "offload_bf16",
        "dropped_context_rows": 0,
        **updates,
    }

    with pytest.raises(RuntimeError, match="forbidden inference memory rung"):
        validate_memory_report(report, dict(NORIDFS_MEMORY_POLICY))


def test_missing_report_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="did not report"):
        validate_memory_report(None, dict(NORIDFS_MEMORY_POLICY))


def test_noridfs_policy_opens_directly_on_host_offload() -> None:
    from synthefy_nori import MemoryPolicy

    policy = MemoryPolicy(**NORIDFS_MEMORY_POLICY)
    resolve = dict(
        est_cache_gb=33.6, bytes_per_element=2, head_dim=16, total_vram_gb=143
    )

    offload = policy.resolve(**resolve, total_ram_gb=128, cache_eligible=True)
    uncached = policy.resolve(**resolve, total_ram_gb=128, cache_eligible=False)

    assert offload.rung == "offload_bf16"
    assert uncached.rung == "no_cache"
