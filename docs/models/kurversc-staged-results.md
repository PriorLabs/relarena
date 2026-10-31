# KurveRSC revised pilot results

Completed: 6/6. Profile: `kurversc-generic-staged-v1`.

| Task | Metric | Revised | Oct 27 pipeline | Captured client | Generic admission | Peak RSS GiB |
|---|---|---:|---:|---:|---|---:|
| rel-f1/driver-dnf | roc_auc | 0.737247 | 0.740482 | 0.727809 | incumbent | 8.84 |
| rel-f1/driver-position | mae | 3.903188 | 3.960206 | 3.757398 | incumbent | 7.29 |
| rel-avito/user-clicks | roc_auc | 0.641794 | 0.647785 | 0.691631 | incumbent | 20.76 |
| rel-trial/study-adverse | mae | 34.365409 | 35.212072 | 34.401374 | zero_aware_1 | 21.71 |
| rel-trial/study-outcome | roc_auc | 0.758745 | 0.755106 | 0.764556 | text50 | 18.38 |
| rel-stack/user-engagement | roc_auc | 0.904074 | 0.907724 | 0.907662 | incumbent | 32.18 |

Scores use RelBench EntityTask.evaluate, scored separately after keyed predictions and validation selection are saved.
Comparators come from the captured October 27 panel, not a new client evaluation. The historical pipeline also used different temporal boundaries.
The first tasks needed two documented metadata/resume repairs; this pilot is not a claim of a clean first-attempt full-harness reproduction.
No revised 21-task Elo is claimed. The other fifteen tasks are held.

[Original October 27 measurements](archive/kurversc-staged-20261027/kurversc-staged-results.md) remain archived.
