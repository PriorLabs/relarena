# KurveRSC resource qualification

## Current frozen source: two completed F1 tasks

At `f5bf98a`, fresh current-harness author runs on the same 48 GiB Mac completed
top3 in 2.709 hours at 11.442 GiB sampled process-tree RSS, and position in
3.135 hours at 10.025 GiB. Both used the unchanged 23.75-hour deadline and
finished without a memory/deadline interruption. Minimum system availability
was 17.906 and 18.965 GiB respectively.

These measurements qualify two tasks only. They do not certify full-panel RAM
or runtime requirements. [Exact receipts and measurement definitions](kurversc-current-harness-f1-qualification.md)
include the existing numerical environment and official scoring provenance.

## Pre-correction narrowed pilot resources

Measured on source `b2cbeea0c7ee26d22e7076c7424c7ab108b2d060`, before the
source-graph replay correction. These observations are planning context for the
current reproduction candidate.

48 GB Mac, one task at a time, 8 GiB RAM reserve. RSS is sampled process-tree RSS, not kernel cgroup memory.peak.

| Task | Status | Wall hours | Recorded phase hours | Peak RSS GiB |
|---|---|---:|---:|---:|
| rel-f1/driver-dnf | completed | 4.42 | 2.45 | 8.84 |
| rel-f1/driver-position | completed | 18.85 | 3.79 | 7.29 |
| rel-avito/user-clicks | completed | 17.50 | 0.48 | 20.76 |
| rel-trial/study-adverse | completed | 10.63 | 10.62 | 21.71 |
| rel-trial/study-outcome | completed | 1.22 | 1.22 | 18.38 |
| rel-stack/user-engagement | completed | 2.44 | 2.44 | 32.18 |

Wall time includes host sleep, waiting and metadata/resume repairs. Recorded phase time can exclude host sleep; it is not a controlled runtime comparison. Each task retains its original deadline below 24 hours.

The other fifteen tasks are held. Full-panel resource requirements for this revision have not yet been measured. The [historical 496-GiB host report](archive/kurversc-staged-20261027/kurversc-staged-resources.md) is retained for context, not assigned to this revision.

Independent tasks can run concurrently with separate output/scratch directories. Keep the per-task worker/thread settings fixed for reproduction, reserve memory headroom, and admit tasks using observed whole-process demand. A DuckDB working-memory threshold governs spilling; it is not the total task RAM requirement.
