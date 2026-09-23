"""Exact inference memory policies and the report check that enforces them."""

from __future__ import annotations

from typing import Any, Final

CONTEXT_ELEMENTS_BUDGET: Final = 3_000_000
HOST_CACHE_BUDGET_GB: Final = 64.0
NORIDFS_CONTEXT_ROW_CHUNK: Final = 512
#: Host-offloaded BF16 cache with no lossy fallback. The report check below
#: rejects any run that Nori quietly escalated to a lossy rung. The plain
#: chunked loop is allowed: it recomputes the context K/V per query chunk,
#: which is slower but bit-exact, and some tasks need it once the estimated
#: cache passes the host budget.
NORIDFS_MEMORY_POLICY: Final = {
    "allow_quantization": False,
    "allow_subsample": False,
    "cache_dtype": "bf16",
    "context_row_chunk": NORIDFS_CONTEXT_ROW_CHUNK,
    "elements_budget": CONTEXT_ELEMENTS_BUDGET,
    "gpu_budget_absolute_gb": 0.0,
    "host_budget_absolute_gb": HOST_CACHE_BUDGET_GB,
    "offload_to_host": True,
    "reuse_context_cache": False,
}
_INVARIANT_KEYS: Final = frozenset(
    {
        "allow_quantization",
        "allow_subsample",
        "cache_dtype",
        "context_row_chunk",
        "elements_budget",
        "reuse_context_cache",
    }
)
_BUDGET_KEYS: Final = frozenset({"gpu_budget_absolute_gb", "host_budget_absolute_gb"})


def validate_memory_report(
    report: dict[str, Any] | None, requested: dict[str, Any]
) -> dict[str, str | int]:
    """Reject lossy or OOM-triggered inference rungs after a prediction.

    ``no_cache``, ``offload_bf16``, and ``plain_loop`` are exact; a GPU budget
    also admits ``resident_bf16``. Anything quantized, subsampled, or that
    dropped context rows fails.
    """
    if report is None:
        raise RuntimeError("nori did not report its inference memory rung")
    from synthefy_nori import MemoryPolicy

    policy = MemoryPolicy(**report)
    resolved = policy.model_dump()
    invariant = set(_INVARIANT_KEYS)
    if policy.rung not in {"no_cache", "plain_loop"}:
        invariant |= _BUDGET_KEYS
    drift = {
        key: (value, resolved.get(key))
        for key, value in requested.items()
        if key in invariant and resolved.get(key) != value
    }
    allowed = {"no_cache", "offload_bf16", "plain_loop"}
    if float(requested["gpu_budget_absolute_gb"]) > 0:
        allowed.add("resident_bf16")
    if (
        policy.rung not in allowed
        or not policy.is_resolved
        or not policy.is_bit_exact
        or policy.cache_dtype != "bf16"
        or policy.dropped_context_rows != 0
        or drift
        or (policy.rung == "offload_bf16" and not policy.offload_to_host)
    ):
        raise RuntimeError(
            "nori used a forbidden inference memory rung: "
            f"rung={policy.rung!r}, cache_dtype={policy.cache_dtype!r}, "
            f"dropped_context_rows={policy.dropped_context_rows}, drift={drift}"
        )
    return {
        "memory_rung_actual": str(policy.rung),
        "memory_cache_dtype_actual": str(policy.cache_dtype),
        "memory_dropped_context_rows": int(policy.dropped_context_rows),
    }


__all__ = [
    "CONTEXT_ELEMENTS_BUDGET",
    "NORIDFS_MEMORY_POLICY",
    "validate_memory_report",
]
