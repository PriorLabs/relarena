# KurveRSC complete staged reproduction: resources

Snapshot: 2026-10-27T13:34:35.809800+00:00. **21/21 tasks completed** in the fresh
installed-wheel campaign. All task resource receipts are finalized, with no
failed tasks or historical score substitutions. The final performance/Elo
table is in the [results report](kurversc-staged-results.md).
These are measured resources, not hypothetical scores or minimum-hardware claims.

The observed host has **495.77 GiB RAM, 64 logical CPUs
(32 physical cores), and a 992.2-GiB filesystem**.
The pipeline uses CPU; no GPU was used. The predictive recipe fixes eight
learner threads per task, graph-search parallelism 1 and temporal-seed workers 1.

## Per-task measurements

| Task | Status | Hours | Peak RAM GiB | Peak tree RSS GiB | Peak busy CPUs | Task disk GiB |
|---|---|---:|---:|---:|---:|---:|
| rel-amazon/item-churn | completed | 3.25 | 132.87 | 123.53 | 8.03 | 21.18 |
| rel-amazon/item-ltv | completed | 5.68 | 182.17 | 161.33 | 8.03 | 22.52 |
| rel-amazon/user-churn | completed | 3.65 | 203.72 | 194.17 | 8.07 | 21.64 |
| rel-amazon/user-ltv | completed | 5.60 | 152.40 | 143.01 | 10.52 | 21.02 |
| rel-avito/ad-ctr | completed | 0.72 | 67.24 | 66.71 | 9.09 | 1.43 |
| rel-avito/user-clicks | completed | 1.14 | 52.03 | 51.42 | 8.69 | 1.59 |
| rel-avito/user-visits | completed | 1.75 | 89.02 | 88.37 | 8.62 | 1.55 |
| rel-event/user-attendance | completed | 2.10 | 116.77 | 116.20 | 8.02 | 0.55 |
| rel-event/user-ignore | completed | 1.45 | 33.04 | 31.37 | 8.06 | 0.70 |
| rel-event/user-repeat | completed | 1.58 | 32.54 | 32.69 | 8.03 | 0.58 |
| rel-f1/driver-dnf | completed | 5.68 | 6.70 | 6.98 | 8.06 | 0.24 |
| rel-f1/driver-position | completed | 4.61 | 4.14 | 4.51 | 8.03 | 0.19 |
| rel-f1/driver-top3 | completed | 3.55 | 2.88 | 3.34 | 8.04 | 0.10 |
| rel-hm/item-sales | completed | 4.90 | 26.53 | 24.40 | 8.03 | 2.10 |
| rel-hm/user-churn | completed | 3.39 | 44.95 | 44.87 | 8.06 | 4.08 |
| rel-stack/post-votes | completed | 5.64 | 61.63 | 58.24 | 11.90 | 11.80 |
| rel-stack/user-badge | completed | 14.17 | 333.51 | 319.19 | 8.01 | 13.35 |
| rel-stack/user-engagement | completed | 4.82 | 91.00 | 86.89 | 8.05 | 11.52 |
| rel-trial/site-success | completed | 4.37 | 91.67 | 89.96 | 8.05 | 2.50 |
| rel-trial/study-adverse | completed | 8.94 | 26.09 | 24.54 | 8.02 | 2.63 |
| rel-trial/study-outcome | completed | 2.69 | 26.90 | 25.54 | 8.03 | 2.65 |

The [CSV](kurversc-staged-resources.csv) preserves full precision and coverage.
RAM is the full-task kernel cgroup-v2 `memory.peak`, including descendants,
charged file cache and kernel memory. Tree RSS is sampled once per second and
can count shared pages more than once. Busy CPUs are sampled cgroup CPU-time
increments per wall second; thread reservations are not affinity limits.
Task disk is sampled allocated storage for owned outputs, temporary files,
logs and metrics, excluding shared datasets, model weights and the environment.
Allow additional storage for those shared inputs and for concurrent tasks.

The CPU/disk observer attached after user-badge had started. Its earlier CPU
and disk peaks are unknown; its full-task cgroup RAM peak and elapsed time are
recorded by the original task monitor. Other attachment delays are in the CSV.
Sampled disk/CPU values can miss short peaks.

## Parallel reproduction on a larger machine

Tasks are independent. Run separate `(dataset, task)` CLI processes or batch
jobs with distinct result files, scratch directories and task-owned caches.
The original absolute sub-24-hour budget applies independently to every task,
including preprocessing, selection, refits and prediction.

Preserve **eight learner threads, one graph-search worker and one temporal-seed
worker per task** for the qualified recipe. Parallelize across tasks to use a
larger host. Increasing internal search concurrency is not part of this measured
configuration and can multiply wide SQL/materialization memory.

Use admission based on future task demand, rather than today's low RSS. One
conservative planning rule is:

```text
task reservation GiB = max(16, ceil(1.25 * max(measured cgroup peak, tree RSS peak) + 8))
host reserve GiB     = max(64, ceil(0.10 * host RAM GiB))
admit only if:
    sum(task reservations) <= host RAM - host reserve
    8 * active task count <= available logical CPUs
    simultaneous scratch/output demand fits disk with headroom
```

For example, a **1.5-TiB (1,536-GiB), 192-logical-CPU** host has a 1,382-GiB
task reservation budget under that rule. It can run several large tasks plus
smaller tasks concurrently when their reservations fit. A multi-node scheduler
can bin-pack the same independent tasks across nodes. A fixed all-21 launch
without checking the reservation sum is not the suggested reproduction plan.

For the measured campaign, the declared DuckDB working-memory ceilings were
123 GiB for 19 tasks, 224 GiB for user-badge and 256 GiB for Amazon user-churn.
The CSV records these resource allocations. Set `KURVERSC_DUCKDB_MEMORY_LIMIT`
explicitly to reproduce a measured allocation on comparable hardware; the
public entry point preserves this resource override. These settings govern
aggregation/spilling and can affect runtime and observed RAM. They do not
reduce source rows, feature budgets or validation evidence.

DuckDB working-memory limits control one engine's aggregation/spilling; they
are not total-process RSS caps. The measured campaign has no hard per-task RAM
kill ceiling. On the measured roughly 496-GiB host, the final large tasks run
sequentially because their conservative reservations do not fit together.

For a straightforward first reproduction, use a comparable 512-GiB-class
machine and retain headroom above the largest observed task. This is the
measured deployment class, not proof that smaller hardware cannot run it.
