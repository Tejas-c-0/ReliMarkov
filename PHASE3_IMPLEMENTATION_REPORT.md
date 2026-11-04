# Phase 3 Validation Implementation Report

## Scope and safeguards

The Phase 3 runner evaluates train-derived models against validation transitions only. It loads the existing train and validation transition files for the 30-minute and 60-minute gaps; it does not read test transitions or test outcomes. Existing splits, configuration, source scripts, and historical experiment outputs were left unchanged. New results were written to:

`outputs/phase3_validation/run_20261103_213050_674910100/`

The earlier Phase 3 output directories were preserved. In particular, the earlier run containing D2 results is not used in this report because its duration metadata coverage was insufficient.

## Files created and outputs

- `kr_experiments_phase3.py` — separate validation-only implementation.
- `test_phase3_validation.py` — nine focused unit tests.
- `outputs/phase3_validation/run_20261103_213050_674910100/validation_metrics_30min.csv`
- `outputs/phase3_validation/run_20261103_213050_674910100/validation_metrics_60min.csv`
- `outputs/phase3_validation/run_20261103_213050_674910100/ranking_overlap_30min.csv`
- `outputs/phase3_validation/run_20261103_213050_674910100/ranking_overlap_60min.csv`
- `outputs/phase3_validation/run_20261103_213050_674910100/lookup_diagnostics_30min.json`
- `outputs/phase3_validation/run_20261103_213050_674910100/lookup_diagnostics_60min.json`
- Run metadata JSON files in that same directory record runtime, memory, validation-case counts, and data scope.

No existing files were modified.

## Implemented equations

For a transition from source `s` to candidate `v`, let `C(s,v)` be the integer transition count, `Wj(s,v)` the corresponding reliability-weighted transition sum, `r_q` the query-source reliability, and `m_v` the training-popularity prior mass. The implementation uses:

- Reliability: `r = clip(watch_ratio + 0.3*is_like + 0.2*is_forward - 0.5*is_hate, 0.1, 1.5)`.
- M4-W1: `C(s,v) + m_v`. It does not use query reliability.
- M4-W2: `r_q * W2(s,v) + m_v`, where `W2` sums source-event reliability.
- M4-W3: `r_q * W3(s,v) + m_v`, where `W3` sums target-event reliability.
- M4-W4: `r_q * W4(s,v) + m_v`, where each transition contributes `sqrt(r_source * r_target)` to `W4`.
- M5-A: `r_q * W4(s,v) + alpha_EB * m_v`.
- M5-B: `r_q * (W4(s,v) / mu) + alpha_EB * m_v`.
- M5-C: `r_q * W4(s,v) + (mu * alpha_EB) * m_v`.

Here `mu` is the mean W4 contribution per training transition, estimated separately for each duration treatment from training data. The loaded standard integer-count EB concentrations are `318.86704207003663` (30-minute gap) and `351.65184635402005` (60-minute gap). They are not refit on reliability-weighted pseudo-counts. M5-C remains a separately named scale-conversion variant; dividing its score by positive `mu` gives M5-B, so their rankings are mathematically scale-equivalent.

For positive event durations, `watch_ratio = clip(play_time_ms / duration_ms, 0, 1)`. For nonpositive durations, D0 uses `0.0` and D1 uses `0.9016` before the shared reliability formula and clipping. The implementation does not introduce a fallback for missing event joins: it counts missing lookups and, if any exist, uses the explicitly documented mean reliability of observed training-transition endpoints. In this run the measured missing lookup counts were zero.

## Validation results

Each gap evaluated 8 models (M1, M4-W1 through M4-W4, and M5-A/B/C) under D0 and D1. The table shows all-subset aggregate results; the complete CSVs contain the exact values, confidence intervals, pool-size summaries, coverage, author-diversity summaries, and all four subsets (`all`, `target_known`, `source_known`, `source_unknown`).

| Gap | Validation cases | Treatment | Hit@10 | Hit@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 30 min | 15,047 | D0 and D1 | 0.00006646 | 0.0001329 | 0.00003323 | 0.00004917 | 0.00002215 | 0.00002606 |
| 60 min | 15,613 | D0 and D1 | 0.00006405 | 0.0001281 | 0.00003202 | 0.00004738 | 0.00002135 | 0.00002512 |

At the displayed precision, the above all-subset ranking metrics are the same for all eight model variants and both D0/D1 treatments within a given gap. This result should not be read as evidence of superiority: these metrics reflect very few hits, and rankings are highly overlapping. The full CSVs retain unrounded values and user-cluster bootstrap 95% intervals (1,000 resamples).

### Candidate pool and coverage

The pre-ranking candidate pool is the source-specific candidate set unioned with the training global-popularity backfill. Across variants, mean candidate-pool size is approximately 50.04 for M4/M5 models and approximately 100 for M1, with the exact model/treatment statistics and 95th percentiles in the metrics CSVs. The recommendation catalogue has 948,911 videos. Overall recommendation coverage is approximately 0.00059–0.00063 at 30 minutes and 0.00064–0.00068 at 60 minutes, varying slightly by model and treatment. Mean distinct authors among Top-10 recommendations is approximately 10; the author-diversity columns are populated from available video-author metadata.

### Known/unknown source and target subsets

For D0/M5-B, illustrative 30-minute results are:

| Subset | Cases | Hit@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Known source | 510 | 0 | 0 | 0 |
| Unknown source | 14,537 | 0.00006879 | 0.00003439 | 0.00002293 |
| Known target | 2,936 | 0.0003406 | 0.0001703 | 0.0001135 |

For D0/M5-B at 60 minutes, the corresponding subset sizes are 549, 15,064, and 3,028; Hit@10 is 0 for known-source, 0.00006638 for unknown-source, and 0.0003303 for known-target. Full metrics for each model, treatment, and subset are in the CSVs.

## Ranking diagnostics

Mean per-query Top-K overlap is the intersection count divided by K; Jaccard is also reported. For D0 at both gaps, M1 vs M4-W1 and M1 vs M4-W4 have Top-10 and Top-20 overlap of 1.0. M1 vs M5-B has overlap of 0.999900/0.999960 (Top-10/20) at 30 minutes and 0.999853/0.999942 at 60 minutes.

For duration sensitivity, M4-W4 has D0-vs-D1 overlap of 1.0 at both K values and gaps. M5-B overlap is 0.999907/0.999963 at 30 minutes and 0.999936/0.999955 at 60 minutes (Top-10/20). Jaccard values and the other requested M1/M4/M5 pairings are recorded in the ranking-overlap CSVs.

## Zero-duration sensitivity and D2

D0 and D1 changed the training mean W4 and the fallback mean reliability:

| Gap | Treatment | Training endpoint fallback mean | Training mean W4 |
|---|---|---:|---:|
| 30 min | D0 | 0.594251 | 0.521290 |
| 30 min | D1 | 0.650259 | 0.584407 |
| 60 min | D0 | 0.594681 | 0.521615 |
| 60 min | D1 | 0.651458 | 0.585613 |

Despite these changes, D0/D1 ranking overlap and all-subset validation metrics are nearly/at identical precision in the emitted summaries. This indicates low observed ranking sensitivity to these two treatments in this experiment; it does not establish that either imputation is preferable.

D2 was not evaluated. Its documented duration-bin thresholds are 7,000, 18,000, and 60,000 ms; training-only duration-bin medians were `1.0`, `1.0`, `0.9020297393`, and `0.3035706085`, with training global median `0.9000582242`. However, video-duration metadata was missing for 8,073 of 8,075 relevant distinct zero-duration source/transition events at 30 minutes and 8,422 of 8,424 at 60 minutes. Applying a duration-conditioned treatment with that coverage would not be a defensible comparison, so the runner explicitly skipped it.

## Tests, runtime, and memory

- `python -m unittest -v test_phase3_validation.py`: **9 tests passed**.
- `python -m py_compile kr_experiments_phase3.py test_phase3_validation.py`: **passed**.
- 30-minute gap: 83.30 seconds; peak working set 1,001.96 MB.
- 60-minute gap: 85.16 seconds; peak working set 1,034.59 MB.
- Total measured experiment time: approximately 168.47 seconds; maximum reported peak working set: approximately 1.01 GiB.

The implementation reads event logs in chunks and only materializes relevant train/validation event keys. Measured memory remained well below the 16 GB constraint.

## Remaining limitations

- D2 could not be evaluated because required video-duration metadata was unavailable for almost all relevant zero-duration events.
- Validation ranking metrics contain very few hits; the resulting comparisons have limited resolution. Interpret bootstrap intervals and subgroup counts alongside point estimates.
- Top-K overlap is near one across the primary models, so the observed validation data show little separation in recommendation lists.
- Author diversity depends on the available author mapping and counts only non-null author IDs.
- The standard EB concentrations were loaded as supplied for integer counts and kept separate from reliability-weighted sums, as approved; this phase does not establish a weighted-likelihood EB estimator.
- No frozen-test evaluation, test outcome inspection, or test metric generation was performed.
