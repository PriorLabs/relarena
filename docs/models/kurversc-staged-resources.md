# KurveRSC resource planning

## Maintainer reproduction

The 21-task reproduction ran one task per host with kurversc 0.2.0. Eight tasks
ran on 96-vCPU, 739 GB hosts with `KURVERSC_DUCKDB_MEMORY_LIMIT=160GiB` and
`KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE=100GiB`. The other 13 ran on 48-core,
190 GB hosts with `48GiB` and `20GiB`. Peak RSS is the Slurm `MaxRSS` of the
job; it grows with the DuckDB limit, so it is an observation under these
limits, not a lower bound.

| Task | Host RAM, GB | Wall time, h | Peak RSS, GiB |
|---|---:|---:|---:|
| rel-amazon/item-churn | 739 | 3.32 | 364 |
| rel-amazon/item-ltv | 739 | 6.48 | 202 |
| rel-amazon/user-churn | 739 | 4.98 | 262 |
| rel-amazon/user-ltv | 739 | 8.86 | 327 |
| rel-avito/ad-ctr | 190 | 0.71 | 34 |
| rel-avito/user-clicks | 190 | 1.21 | 51 |
| rel-avito/user-visits | 190 | 1.14 | 64 |
| rel-event/user-attendance | 190 | 3.35 | 46 |
| rel-event/user-ignore | 739 | 1.83 | 283 |
| rel-event/user-repeat | 190 | 1.72 | 44 |
| rel-f1/driver-dnf | 190 | 5.88 | 6 |
| rel-f1/driver-position | 190 | 4.88 | 4 |
| rel-f1/driver-top3 | 190 | 3.93 | 4 |
| rel-hm/item-sales | 190 | 3.47 | 27 |
| rel-hm/user-churn | 190 | 1.64 | 21 |
| rel-stack/post-votes | 739 | 10.60 | 257 |
| rel-stack/user-badge | 739 | 10.38 | 370 |
| rel-stack/user-engagement | 739 | 4.36 | 219 |
| rel-trial/site-success | 190 | 1.73 | 89 |
| rel-trial/study-adverse | 190 | 6.36 | 30 |
| rel-trial/study-outcome | 190 | 2.27 | 23 |

rel-event/user-ignore and rel-stack/user-engagement ran out of memory on the
190 GB hosts with the 48GiB DuckDB limit, and rel-stack/post-votes reached
140 GiB there before it was moved; all three completed on the 739 GB hosts.

## Author measurements

The following author measurements are planning context from explicitly older
source revisions, not resource guarantees for the new all-task S2 check.

| Source | Task | Sampled peak process-tree RSS, GiB |
|---|---|---:|
| `f5bf98a` | rel-f1/driver-top3 | 11.44 |
| `f5bf98a` | rel-f1/driver-position | 10.03 |
| `b2cbeea` | rel-f1/driver-dnf | 8.84 |
| `b2cbeea` | rel-avito/user-clicks | 20.76 |
| `b2cbeea` | rel-trial/study-adverse | 21.71 |
| `b2cbeea` | rel-trial/study-outcome | 18.38 |
| `b2cbeea` | rel-stack/user-engagement | 32.18 |

These tasks ran on a 48 GiB Mac, sequentially with an 8 GiB reserve. The two
`f5bf98a` tasks took 2.71/3.14 hours. The earlier pilot included host sleep and
documented repairs, so its wall times are not controlled runtime estimates.
Complete source receipts remain in the prior PR history and KurveRSC evidence.

DuckDB working-memory limits control spilling; they do not bound total task
memory. The per-task prediction procedure keeps its original deadline below
24 hours and raises on incomplete execution. Use separate output/scratch/cache
directories for concurrent tasks, retain system headroom, and preserve the
submitted internal worker/thread settings. A large host may parallelize tasks
based on measured whole-process demand; these seven observations do not
establish peak requirements for the other tasks.
