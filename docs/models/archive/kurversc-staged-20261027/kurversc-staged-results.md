> Historical October 27 recipe, superseded after maintainer review. These measurements do not describe the revised submission.

# Complete staged KurveRSC: final fresh results

All 21 tasks completed under one frozen payload with no task failures or
historical score fills. The official scorer replaces the original KurveRSC row
in the captured 15-method panel: **1948.6 Elo, rank 1**, versus **1937.1** for
`tabpfn-rel-client-2026-09-28`; the point lead is **11.5**, with overlapping
task-bootstrap intervals. Head-to-head results are **11 wins / 10 losses**.
Mean rescaled loss is **0.073390** versus **0.073884** (lower is better).

This is the completed author-run comparison on the October 25 captured panel,
not an accepted public-leaderboard update or independent maintainer reproduction.
The score-producing wheels carry private development labels 0.1.1/1.10.17;
the 0.2.0/1.10.18 review payload matches before version stamping. Current core
split/classification/regression smoke checks preserve keyed predictions exactly.

| Task | Metric | Previous KurveRSC | Fresh KurveRSC | TabPFN client | Winner |
| --- | --- | --- | --- | --- | --- |
| Amazon / item-churn | AUC | 0.827256 | 0.829162 | 0.828653 | Kurve |
| Amazon / item-ltv | MAE | 42.273080 | 42.575857 | 45.270515 | Kurve |
| Amazon / user-churn | AUC | 0.710347 | 0.711315 | 0.711698 | Client |
| Amazon / user-ltv | MAE | 14.020545 | 14.005336 | 13.877122 | Client |
| Avito / ad-ctr | MAE | 0.03344026 | 0.03281736 | 0.03125771 | Client |
| Avito / user-clicks | AUC | 0.658406 | 0.647785 | 0.691631 | Client |
| Avito / user-visits | AUC | 0.674721 | 0.673024 | 0.672453 | Kurve |
| Event / user-attendance | MAE | 0.25866836 | 0.23748723 | 0.24165168 | Kurve |
| Event / user-ignore | AUC | 0.832628 | 0.843941 | 0.784037 | Kurve |
| Event / user-repeat | AUC | 0.779679 | 0.806016 | 0.760495 | Kurve |
| F1 / driver-dnf | AUC | 0.746484 | 0.740482 | 0.727809 | Kurve |
| F1 / driver-position | MAE | 3.863205 | 3.960206 | 3.757398 | Client |
| F1 / driver-top3 | AUC | 0.664957 | 0.803766 | 0.791702 | Kurve |
| H&M / item-sales | MAE | 0.03216765 | 0.03173097 | 0.03647878 | Kurve |
| H&M / user-churn | AUC | 0.696916 | 0.711248 | 0.706021 | Kurve |
| Stack / post-votes | MAE | 0.06324286 | 0.06273966 | 0.06273299 | Client |
| Stack / user-badge | AUC | 0.877626 | 0.877817 | 0.883443 | Client |
| Stack / user-engagement | AUC | 0.903027 | 0.907724 | 0.907662 | Kurve |
| Trial / site-success | MAE | 0.40384204 | 0.39135221 | 0.37761499 | Client |
| Trial / study-adverse | MAE | 41.270229 | 35.212072 | 34.401374 | Client |
| Trial / study-outcome | AUC | 0.712131 | 0.755106 | 0.764556 | Client |

Values above are rounded; post-votes remains a loss at full precision. The
[CSV](kurversc-staged-results.csv) contains full precision and artifact hashes.
The [resource report](kurversc-staged-resources.md) records every task's runtime,
peak RAM, CPU/storage coverage and safe opportunities for parallel execution.


For the alternative reporting scope that retains the historical native
KurveRSC and adds this dated system as a sixteenth method, the same official
scorer gives **1954.2 Elo (rank 1)** versus **1940.1** for the client. The
15-method replacement comparison above remains the primary like-for-like
scorecard. Both panels use all 21 fresh scores and the same captured competitor
results; public acceptance and independent reproduction remain separate.

Sixteen tasks improve over the captured previous KurveRSC scores; five decline.
An incumbent gate is a selection safeguard, not a universal no-regression claim.
The full pipeline remains subject to independent maintainer reproduction and
the external pretrained-encoder policy review described in the integration docs.
