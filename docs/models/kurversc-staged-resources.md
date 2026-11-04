# KurveRSC resource planning

Full-panel resource measurements for the revised reproduction pin are pending.
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
