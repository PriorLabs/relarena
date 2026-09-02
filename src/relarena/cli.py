"""Command-line entry point: run a registered model across relational tasks on RelBench.

Examples:
--------
    # list the RelBench v1 entity tasks that would run (no download):
    relarena --list

    # run the constant baseline on all v1 datasets, write results to a CSV:
    relarena --model constant-global --output results.csv

    # a single dataset / task:
    relarena --datasets rel-f1 --tasks driver-dnf

Note: actually running downloads each dataset (several GB for the full v1 set).
Use `--list` first to preview the plan without downloading anything.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import relarena.models  # noqa: F401  (registers built-in models)
from relarena.registry import registry
from relarena.results import summary_to_dataframe
from relarena.runner import SystemExperimentSummary, run_experiment
from relarena.tasks import RELBENCH_V1_DATASETS, list_entity_tasks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relarena", description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        default=None,
        help="registered model name (required, unless --list); e.g. 'constant-global'",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=list(RELBENCH_V1_DATASETS),
        help="dataset names to run (default: the 7 RelBench v1 datasets)",
    )
    p.add_argument(
        "--tasks", nargs="*", default=None, help="restrict to these task names"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="HPO trials (ignored by untunable models)",
    )
    p.add_argument("--no-test", action="store_true", help="skip test-set evaluation")
    p.add_argument(
        "--list",
        action="store_true",
        help="list discovered tasks and exit (no download)",
    )
    p.add_argument("--output", default=None, help="write a results CSV to this path")
    p.add_argument("--cache-dir", default=None, help="local preprocessing cache")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CLI: discover tasks, run the model on each, print/write results."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.model is None and not args.list:
        parser.error(
            "--model is required (e.g. --model constant-global). "
            "Use --list to preview tasks."
        )

    specs = list_entity_tasks(args.datasets)
    if args.tasks:
        wanted = set(args.tasks)
        specs = [s for s in specs if s.task in wanted]

    n_ds = len({s.dataset for s in specs})
    print(f"Discovered {len(specs)} entity task(s) across {n_ds} dataset(s):")
    for s in specs:
        print(f"  {s.dataset:12s} {s.task:28s} {s.task_type.name}")
    if args.list:
        return 0
    if not specs:
        print("Nothing to run.", file=sys.stderr)
        return 1

    model_cls = registry.get(args.model)
    frames: list[pd.DataFrame] = []
    for s in specs:
        print(
            f"\n=== {args.model} on {s.dataset}/{s.task} ({s.task_type.name}) ===",
            flush=True,
        )
        try:
            summary = run_experiment(
                model_cls,
                s.dataset,
                s.task,
                seed=args.seed,
                n_trials=args.n_trials,
                cache_dir=args.cache_dir,
                evaluate_test=not args.no_test,
            )
        except Exception as exc:  # one bad dataset shouldn't abort the sweep
            print(f"    ERROR: {exc!r}", file=sys.stderr)
            continue
        frames.append(summary_to_dataframe(summary))
        if isinstance(summary, SystemExperimentSummary):
            score = summary.result.test_score
            score_str = f"{score:.4f}" if score is not None else "n/a"
            print(f"    {summary.metric_name}: test={score_str}")
        else:
            best = summary.tuned or summary.default
            if best is None or best.val_score is None:
                print("    (no successful trial)")
            else:
                test_str = (
                    f"{best.test_score:.4f}" if best.test_score is not None else "n/a"
                )
                print(
                    f"    {summary.metric_name}: val={best.val_score:.4f} "
                    f"test={test_str}"
                )

    if not frames:
        print("No successful runs.", file=sys.stderr)
        return 1

    results = pd.concat(frames, ignore_index=True)
    selected = results[results["selected"]].reset_index(drop=True)
    print("\n==== RESULTS (val-selected config per task) ====")
    print(selected.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nWrote {args.output} ({len(results)} rows, all configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
