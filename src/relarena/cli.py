"""Command-line entry point: run a registered method across RelBench tasks.

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
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import relarena.models  # noqa: F401  (registers built-in methods)
from relarena.registry import registry
from relarena.results import summary_to_dataframe
from relarena.runner import SystemExperimentSummary, run_experiment
from relarena.tasks import RELBENCH_V1_DATASETS, list_entity_tasks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relarena", description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        default=None,
        help="registered method name (required, unless --list); e.g. 'constant-global'",
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
        help="HPO trials (ignored by untunable models and systems)",
    )
    p.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        metavar="N",
        help=(
            "run up to N independent dataset/task experiments concurrently "
            "(default: 1); trials and frame construction within each task remain "
            "sequential"
        ),
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
    """Run the CLI: discover tasks, run the method, and print/write results."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.parallel_tasks < 1:
        parser.error("--parallel-tasks must be at least 1")
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

    method_cls = registry.get(args.model)
    run_kwargs = {
        "seed": args.seed,
        "n_trials": args.n_trials,
        "cache_dir": args.cache_dir,
        "evaluate_test": not args.no_test,
        # The CLI serializes metrics, not prediction arrays. Keeping them out of
        # worker results avoids copying large arrays between task processes.
        "cache_predictions": False,
    }
    frames_by_index: dict[int, pd.DataFrame] = {}

    def record(index: int, summary: object) -> None:
        frames_by_index[index] = summary_to_dataframe(summary)
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
        if summary.peak_rss_gib is not None:
            print(f"    peak_rss_gib: {summary.peak_rss_gib:.2f}")

    if args.parallel_tasks == 1:
        for index, spec in enumerate(specs):
            print(
                f"\n=== {args.model} on {spec.dataset}/{spec.task} "
                f"({spec.task_type.name}) ===",
                flush=True,
            )
            try:
                summary = run_experiment(
                    method_cls,
                    spec.dataset,
                    spec.task,
                    **run_kwargs,
                )
            except Exception as exc:  # one bad dataset shouldn't abort the sweep
                print(f"    ERROR: {exc!r}", file=sys.stderr)
                continue
            record(index, summary)
    else:
        workers = min(args.parallel_tasks, len(specs))
        print(
            f"\nRunning {len(specs)} tasks with parallel_tasks={workers}. "
            "Each task retains sequential frame construction and trials.",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            # Release native-library state and large relational frames after
            # every task instead of retaining them in a long-lived worker.
            max_tasks_per_child=1,
        ) as executor:
            futures = {
                executor.submit(
                    run_experiment,
                    method_cls,
                    spec.dataset,
                    spec.task,
                    **run_kwargs,
                ): (index, spec)
                for index, spec in enumerate(specs)
            }
            completed = 0
            for future in as_completed(futures):
                index, spec = futures[future]
                completed += 1
                print(
                    f"\n=== [{completed}/{len(specs)} completed] {args.model} on "
                    f"{spec.dataset}/{spec.task} ({spec.task_type.name}) ===",
                    flush=True,
                )
                try:
                    summary = future.result()
                except Exception as exc:  # one bad task should not abort the sweep
                    print(f"    ERROR: {exc!r}", file=sys.stderr)
                    continue
                record(index, summary)

    if not frames_by_index:
        print("No successful runs.", file=sys.stderr)
        return 1

    frames = [frames_by_index[index] for index in sorted(frames_by_index)]
    results = pd.concat(frames, ignore_index=True)
    selected = results[results["selected"]].reset_index(drop=True)
    print("\n==== RESULTS (selected row per task) ====")
    print(selected.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nWrote {args.output} ({len(results)} rows, all configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
