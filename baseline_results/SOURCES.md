# Sources for the `_MR` reference baselines

In the TabPFN-3 model report ([arXiv:2605.13986](https://arxiv.org/abs/2605.13986)),
where TabPFN-Rel was first introduced, we followed Hudovernik et al. ([arXiv:2604.12596](https://arxiv.org/abs/2604.12596)) and
copied baseline values from the underlying works directly instead of
re-running them. The only exceptions are **TabPFN-Rel** (our own model),
**RDBLearn** (its paper does not provide numbers for all RelBench datasets),
and **KumoRFM-2** (its own evaluation does not freeze the database at the test
cutoff as set out in RelBench) — these three we ran or re-ran ourselves for
the report. For reference we provide these results as additional results with
the `_MR` (model report) suffix in `reference_results.csv`.

Below is the detailed source for each method and any further comments.

## Per-method sources

| method | `source` token | source | comments |
|---|---|---|---|
| `graphsage_MR` | `relbenchv1_paper` | RelBenchV1 paper ([arXiv:2407.20060](https://arxiv.org/abs/2407.20060)), Tables 3/4, test split | the paper labels the column "RDL"; its Appendix B identifies the model as heterogeneous GraphSAGE |
| `relgnn_MR` | `relgnn_paper` | RelGNN paper ([arXiv:2502.06784](https://arxiv.org/abs/2502.06784)), Tables 1/2, "RelGNN (ours)" column | |
| `relgt_MR` | `relgt_paper` | RelGT paper ([arXiv:2505.10960](https://arxiv.org/abs/2505.10960)), Table 1 | |
| `kumo_rfm_v1_MR` | `kumorfm_v1_whitepaper` | KumoRFM whitepaper (Kumo.AI, 2025; original URL now 404 — [Wayback snapshot](https://web.archive.org/web/20251219072245/https://kumo.ai/research/kumo_relational_foundation_model.pdf)), Tables 2/3, "KumoRFM (in-context)" column | see the rel-avito note below |
| `lightgbm_MR` | `kumorfmv2_report` | KumoRFM-2 report ([arXiv:2604.12596](https://arxiv.org/abs/2604.12596)), Tables 3/7, plain "LightGBM" row | not to be confused with the report's stronger "DS+LightGBM" row |
| `rdblearn_MR` | `tabpfn3_report` | our own run for the TabPFN-3 report | the RDBLearn paper ([arXiv:2602.18495](https://arxiv.org/abs/2602.18495)) omits all rel-f1 tasks and the rel-event classification tasks, and its column mixes backends per task (incl. LimiX), so paper values differ by design |
| `kumo_rfm_v2_MR` | `tabpfn3_report` | our own re-run for the TabPFN-3 report | database frozen at the test cutoff per RelBench, unlike Kumo's own evaluation |
| `tabpfn-rel_MR` | `tabpfn3_report` | our own model's number from the TabPFN-3 report | |

Classification values are AUROC (stored fractionally here; most sources print
×100), regression values are MAE in native units.

## Note: KumoRFM-1 on rel-avito

The sources disagree on which value belongs to which task (AUROC ×100):

| task | KumoRFM v1 whitepaper | KumoRFM-2 model report |
|---|---|---|
| `rel-avito/user-clicks` | 64.11 | 64.85 |
| `rel-avito/user-visits` | 64.85 | 64.11 |

`reference_results.csv` follows the KumoRFM-2 model report (as did the
TabPFN-3 model report); it is unclear to us which assignment is the right one.
The two rel-avito rows carry `source = kumorfmv2_report` accordingly.

## Caveats when citing

- `kumo_rfm_v1_MR` likely follows a different evaluation protocol that
  overestimates performance (the TabPFN-3 report flags it for exactly this);
  the v1 model is deprecated and cannot be re-run.
- Reported numbers carry no tuning sweep, validation scores, or timing; treat
  them as reference points, not reproductions (see `README.md` for how they
  are kept out of the default leaderboard).
- The RDBLearn paper's column reports the per-task best over its backends,
  including LimiX, while relarena's `rdblearn` model sweeps only the TabPFN
  backends as LimiX ships no installable package (see `rdblearn.py`). Therefore,
  the performance of `rdblearn` can likely be improved at the additional cost of
  tuning over an additional TFM.
