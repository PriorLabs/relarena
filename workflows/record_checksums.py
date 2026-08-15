r"""Record / check RelBench v1 split checksums (CLI over `relarena.checksums`).

Slow: loads + hashes every split and censored DB, downloading as needed. Run it
deliberately — record once, and `--check` when bumping RelBench (and from the
bump-gated CI job) — not on every load.

    python workflows/record_checksums.py                      # record the baseline
    python workflows/record_checksums.py --check              # diff vs baseline
    python workflows/record_checksums.py --datasets rel-f1 --tasks driver-dnf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from relarena.checksums import CHECKSUMS_PATH, check_checksums, record_checksums
from relarena.tasks import RELBENCH_V1_DATASETS, list_entity_tasks


def main(argv: list[str] | None = None) -> int:
    """Record or check full-split checksums for the discovered v1 entity tasks."""
    p = argparse.ArgumentParser(prog="record_checksums", description=__doc__)
    p.add_argument("--datasets", nargs="*", default=list(RELBENCH_V1_DATASETS))
    p.add_argument("--tasks", nargs="*", default=None, help="restrict to these tasks")
    p.add_argument("--output", type=Path, default=CHECKSUMS_PATH)
    p.add_argument(
        "--check",
        action="store_true",
        help="compare against the recorded baseline instead of overwriting it; "
        "exit non-zero on any mismatch",
    )
    args = p.parse_args(argv)

    discovered = list_entity_tasks(args.datasets)
    if args.tasks:
        wanted = set(args.tasks)
        discovered = [s for s in discovered if s.task in wanted]
    specs = [(s.dataset, s.task) for s in discovered]

    if not args.check:
        record_checksums(specs, args.output)
        return 0

    mismatches = check_checksums(specs, args.output)
    if not mismatches:
        print(f"\nAll {len(specs)} task checksum(s) match {args.output}.")
        return 0
    print(f"\n{len(mismatches)} task(s) differ from {args.output}:")
    for key, diff in mismatches.items():
        for split, (recorded, computed) in sorted(diff.items()):
            print(f"  {key} [{split}]: recorded={recorded} computed={computed}")
    print(
        "\nIf this is an intentional change (e.g. a deliberate RelBench bump), "
        "re-record with `python workflows/record_checksums.py`."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
