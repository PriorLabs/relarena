# Revised pilot resources

48 GB Mac, one task at a time, 8 GiB RAM reserve. RSS is sampled process-tree RSS, not kernel cgroup memory.peak.

| Task | Status | Wall hours | Recorded phase hours | Peak RSS GiB |
|---|---|---:|---:|---:|
| rel-f1/driver-dnf | completed | 4.42 | 2.45 | 8.84 |
| rel-f1/driver-position | completed | 18.85 | 3.79 | 7.29 |
| rel-avito/user-clicks | completed | 17.50 | 0.48 | 20.76 |
| rel-trial/study-adverse | completed | 10.63 | 10.62 | 21.71 |
| rel-trial/study-outcome | running | — | — | — |
| rel-stack/user-engagement | queued | — | — | — |

Wall time includes host sleep, waiting and metadata/resume repairs. Recorded phase time can exclude host sleep; it is not a controlled runtime comparison. Each task retains its original deadline below 24 hours.

The other fifteen tasks are held. Full-panel resource requirements for this revision have not yet been measured. The [historical 496-GiB host report](archive/kurversc-staged-20261027/kurversc-staged-resources.md) is retained for context, not assigned to this revision.

Independent tasks can run concurrently with separate output/scratch directories. Keep the per-task worker/thread settings fixed for reproduction, reserve memory headroom, and admit tasks using observed whole-process demand. A DuckDB working-memory threshold governs spilling; it is not the total task RAM requirement.
