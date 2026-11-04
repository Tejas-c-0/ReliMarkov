# Phase 4 Results

## Run status

- [OPEN] The full Phase 4 runner was not completed in this session. The user requested that the long command be left to them.
- [OPEN] No Phase 4 metrics, confidence intervals, query-discordance counts, diagnostic values, runtime, or memory measurements are reported below.
- [MEASURED] Phase 3's existing validation outputs report 15,047 queries at 30 minutes and 15,613 at 60 minutes; all evaluated Phase 3 models had one Hit@10 and two Hit@20 hits at each gap.

## Phase 3 comparison baseline

| Gap | Phase | Queries | Hit@10 | Hit@10 95% interval | Hit@20 | Hit@20 95% interval | Candidate pool summary |
|---|---|---:|---:|---:|---:|---:|---|
| 30 min | [MEASURED] Phase 3, M1/D0 | 15,047 | 1/15,047 = 0.00006646 | [0, 0.00020948] | 2/15,047 = 0.00013292 | [0, 0.00034345] | M1 about 100; M4/M5 about 50 |
| 60 min | [MEASURED] Phase 3, M1/D0 | 15,613 | 1/15,613 = 0.00006405 | [0, 0.00020742] | 2/15,613 = 0.00012810 | [0, 0.00032411] | M1 about 100; M4/M5 about 50 |
| 30 min | [OPEN] Phase 4, every model on shared candidates | — | — | — | — | — | Run pending |
| 60 min | [OPEN] Phase 4, every model on shared candidates | — | — | — | — | — | Run pending |

[DERIVED] The Phase 3 fractions above are the reported integer hits divided by the query counts; the displayed decimal rates and intervals are present in `outputs/phase3_validation/run_20261103_213050_674910100/validation_metrics_30min.csv` and `validation_metrics_60min.csv`.

## Required comparisons

- [DERIVED] **Did shared candidates fix pool fairness?** They fix cross-model candidate-pool fairness by construction: each query's source-neighbor union training Top-100 set is constructed once and supplied to each model/treatment. This is a control on the experiment, not evidence that the retrieval pool is exhaustive.
- [OPEN] **Did reliability weighting or smoothing change ranking quality?** Requires Phase 4 metrics and ranking diagnostics.
- [OPEN] **Did any model beat M1?** Requires Phase 4 hit/NDCG/MRR comparisons on the shared pools.
- [OPEN] **Are differences statistically meaningful?** Requires Phase 4 user-cluster bootstrap intervals and model-pair discordant-query counts. No Phase 4 intervals or discordance counts are available yet.
- [MEASURED] **Was catalogue coverage low in Phase 3?** Yes: Phase 3 recommendation coverage was approximately 0.00059–0.00068 against a 948,911-video training catalogue.
- [OPEN] **Does low catalogue coverage remain a limit under Phase 4's shared pool?** Requires Phase 4 recommendation coverage and target-in-candidate-set denominators.
- [OPEN] **Are video-level and author-level findings consistent?** Requires the Phase 4 video and separate next-author results.

## Interpretation boundary

- [OPEN] Do not interpret Phase 3's near-zero hit rates or identical model hit counts as Phase 4 outcomes.
- [OPEN] After the user's run, run `phase4_validity_check.py` before adding measured Phase 4 findings, then run `phase4_extra_diagnostics.py` for each gap.
