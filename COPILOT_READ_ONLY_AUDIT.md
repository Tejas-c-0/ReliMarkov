# Read-Only Project Audit

**Project:** Reliability-Aware Markov Chain Recommendation for Personalised Content  
**Audit scope:** Source scripts, pipeline/evaluation implementations, EB and reliability diagnostics, validation artifacts, configuration files, logs, and reports.  
**Date:** 2026-11-03

No existing files were modified or deleted. No test evaluation was run, and test outcome data was not accessed.

## Verified facts

### Project structure and workflow

The project contains exploratory and legacy scripts (`00_profile.py` through `24_*`), a current chronological pipeline and evaluation code, split/data artifacts, EB parameter JSON files, validation CSVs, and run logs. The principal current scripts are `kr_pipeline.py`, `estimate_eb.py`, `kr_experiments.py`, and `kr_eval.py`. The only Markdown file present before this audit was the dataset `README.md`; no project-specific research report was found.

The current pipeline:

1. Filters standard interaction logs to `tab == 1` and `is_click == 1` (Dataset A).
2. Groups events by user and timestamp. Timestamp groups with multiple events are treated as boundaries; no order is inferred among simultaneous events.
3. Forms transitions only between adjacent valid singleton groups with positive time gap no greater than the configured 30 or 60 minutes.
4. Assigns each transition by its target timestamp: train April 8–28, buffer April 29, validation April 30–May 3, and test May 4–8. Buffer-target transitions are excluded, but a buffer event can provide context for a validation transition.
5. Builds the video catalogue, popularity counts, and author-to-video map from training-period events.

`pipeline_run.txt` and the split configs report 66,876 train / 15,047 validation / 18,476 test transitions for 30 minutes and 69,541 / 15,613 / 19,175 for 60 minutes. The configs report a training catalogue of 948,911 videos in both cases.

### Models implemented

- **M1 — Markov MLE:** ranks observed next-video counts for the source, breaks ties using training popularity, then backfills from global popularity.
- **M2 — fixed Dirichlet:** ranks by `C(s,v) + α m_v` for observed transitions and the 50 most popular training videos. Validation settings use α = 1, 10, and 50.
- **M3 — EB Dirichlet:** uses the M2 score with video-level empirical-Bayes concentration.
- **M4 — reliability-weighted Markov:** uses weighted transition totals and α = 1. W1–W4 are the unweighted, source-only, target-only, and geometric-mean ablations.
- **M5-A — raw:** `w_query * C_W4(s,v) + α_EB m_v`.
- **M5-B — normalized:** `(w_query/μ) * (C_W4(s,v)/μ) + α_EB m_v`, where μ is mean W4 in training.
- **M5-C — calibrated:** `w_query * C_W4(s,v) + λ α_EB m_v`; the validation runner passes the gap-specific μ as λ.

No implementation explicitly named **M5-Ext** was found. The available files do not establish whether M5-C is intended to be M5-Ext.

### EB estimates

`estimate_eb.py` fits separate video- and author-level Dirichlet-multinomial concentrations using training transitions and training popularity as the base measure. It uses bounded scalar optimization over `[0.1, 50000]`.

| Session gap | Video α₀ | Author α₀ |
|---|---:|---:|
| 30 minutes | 318.8670 | 2890.7208 |
| 60 minutes | 351.6518 | 2657.5945 |

These are stored in `outputs/eb_params/eb_hyperparams_30min.json` and `outputs/eb_params/eb_hyperparams_60min.json`.

### W4 and reliability

In `kr_experiments.py`, event reliability is the clipped watch ratio plus `0.3 × is_like + 0.2 × is_forward − 0.5 × is_hate`, clipped to `[0.1, 1.5]`. For zero-duration events the watch ratio is imputed as `0.9016`. Missing event-weight lookups default to `0.68`.

Per-transition W4 is `sqrt(w_src × w_tgt)`. It is accumulated over all training transitions, separately for video and author transitions. The experiment logs report mean W4 of 0.5844 for 30 minutes and 0.5856 for 60 minutes. Thus the reported training mean is over the transition population, not over distinct transition types or source states.

### Validation workflow and metrics

`kr_experiments.py` evaluates its model variants on the validation transitions only. `kr_eval.py` evaluates five baselines; its CLI also offers a test option, but that option was not run during this audit.

Validation metrics include video Hit/Recall@10 and @20, NDCG@10, MRR@10, author Hit/Recall@10 and @20, target-known subset metrics, catalogue/source coverage diagnostics, and user-cluster bootstrap confidence intervals (1,000 resamples, seed 42). Recommendation lists are checked for length and training-catalogue membership.

Video candidate generation uses observed transitions plus the 50 most popular training videos. Author models expand predicted authors into their training videos, then backfill from global video popularity as needed. Validation outputs are in:

- `outputs/results/val_experiments_30min.csv`
- `outputs/results/val_experiments_60min.csv`
- `outputs/results/results_val_30min.csv`
- `outputs/results/results_val_60min.csv`

## Potential bugs or methodological inconsistencies

1. **M5-B scaling:** Both the query weight and accumulated W4 count are divided by μ, so the weighted evidence term is scaled by `1/μ²`. This may not match the stated intention of normalizing W4 to mean 1.
2. **M4 W1:** The W1 branch selects unweighted integer counts, but the shared scoring formula still multiplies the count by the query-event weight.
3. **M5-C label and parameter:** The result label says `calib_0.55`, while the runner passes gap-specific μ (about 0.584/0.586); the function default is 0.5543. The displayed label does not exactly identify the supplied value.
4. **Reliability definitions differ:** `reliability_diagnostics.py` uses `0.5` for zero-duration watch ratio and examines only the first 20,000 training transitions. The main experiment uses `0.9016` and the full training-transition population. Older scripts also contain other formulas, including `long_view`.
5. **EB/count scale:** EB concentrations are fit on integer transition counts and reused with weighted pseudo-counts in M5. This scale assumption is not demonstrated to be equivalent to fitting EB on weighted observations.
6. **Legacy analysis differs from current protocol:** Older scripts create alternative 5-core or quantile-based splits. `21_evaluation_coverage_audit.py` and other legacy scripts include test coverage/evaluation logic; those are not the current date-boundary validation workflow.
7. **Ties:** Some rankers sort candidate sets by score without an explicit deterministic tie-break. Rankings under tied scores may therefore be underspecified.

## Unresolved questions

- Is M5-Ext intended to refer to M5-C, or is a separate model specification missing?
- What is the provenance and intended interpretation of the `0.9016` imputation, `0.68` fallback, and M5-C calibration value?
- Were any legacy test-evaluation scripts actually executed? Test-access code and legacy test-transition artifacts exist, but this audit did not inspect test outcomes.
- The available artifacts do not provide enough version history or execution provenance to establish when each split and result file was created.

## Frozen-test assessment

The current validation logs/results are validation-scoped, and no `results_test*.csv` was found in the output listing. However, the current pipeline writes test-transition artifacts; `kr_eval.py` accepts `--split test`; and legacy scripts explicitly read or evaluate test data. Therefore, the frozen test split's **untouched status cannot be certified from the available logs and artifacts**. This is an evidence limitation, not evidence that the current evaluation ran on test.

## Recommended next actions

1. Clarify and document the intended M5-Ext definition.
2. Resolve and document W1/W4 scoring, M5-B scaling, reliability imputations, and calibration parameter naming before further experiments.
3. Establish frozen-test provenance from independent execution records; continue to keep test evaluation disabled unless its protocol is explicitly approved.
4. Keep any follow-up experimentation limited to validation and preserve historical outputs.
