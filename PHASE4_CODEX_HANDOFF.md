# RELIMARKOV — CODEX PROJECT HANDOFF AND PHASE 4 EXECUTION

This document is the execution brief for Codex when the ReliMarkov project folder is open locally. Work directly in the existing project. Preserve the current experiment design and existing research artifacts.

## 1. Research objective

**Title:** “Markov Chain-Based Recommendation System for Personalised Content: Laplace and Empirical Bayesian Smoothing with a Reliability-Aware Extension, Evaluated on Synthetic Movie Sequences and KuaiRand-1K.”

The research studies Markov-chain-based sequential recommendation, comparing transition-probability estimation and reliability-aware approaches using:
1. Synthetic movie-sequence experiments.
2. The KuaiRand-1K dataset.

The objective is to evaluate whether reliability-aware transition estimation improves recommendation quality relative to the existing baseline approaches.

## 2. Existing KuaiRand experiment design

The evaluation uses two temporal transition gaps:
- 30 minutes.
- 60 minutes.

Chronological split (dataset timestamps use UTC+8):
- Training: April 8–28, 2022.
- Buffer: April 29, 2022.
- Validation: April 30–May 3, 2022.
- Test: May 4–8, 2022.

Treatments:
- **D0:** Existing transition estimation without the reliability adjustment.
- **D1:** Reliability-aware transition estimation.

Model configurations:
- M1
- M4_W1
- M4_W2
- M4_W3
- M4_W4
- M5A
- M5B
- M5C

Evaluation includes Top-10 and Top-20 recommendation metrics, a common candidate set, ranking diagnostics, and cluster bootstrap evaluation. The Phase 4 implementation is expected to use 1,000 bootstrap iterations with seed 42; verify these settings in source before reporting them.

## 3. First task: inspect the connected project folder

Identify the actual locations and status of:
- Main Phase 4 implementation.
- Timed execution wrapper.
- Helper modules imported by Phase 4.
- Phase 4 tests.
- Existing Phase 3 outputs.
- Training and validation Parquet files for both gaps.
- Catalogue, popularity, author-video mapping, and empirical Bayes configuration files.
- Raw log files required by the implementation.
- Python dependencies and environment information.

Expected filenames may include:
- `kr_experiments_phase4.py`
- `phase4_timed_run.py`
- `train_transitions_30min.parquet`
- `val_transitions_30min.parquet`
- `train_transitions_60min.parquet`
- `val_transitions_60min.parquet`

These are clues, not guaranteed paths. Discover actual paths. Do not assume files exist.

## 4. Controlled execution sequence

Do not launch a large experiment blindly. Follow this order.

### Step A — Static inspection
1. Read the Phase 4 implementation, timed wrapper, and directly imported helpers.
2. Trace the actual execution path, required inputs, dependencies, output locations, and likely failure points.
3. Verify validation does not consume test-period events.
4. Verify D0/D1 differ only in the intended reliability treatment.
5. Verify candidate-set consistency and metric denominators.
6. Check whether outputs could overwrite existing results.

Do not refactor working code or rewrite the project just for style.

### Step B — Focused tests
Run the existing Phase 4 tests once. If something fails, diagnose the specific cause, make the smallest justified correction, and rerun only the affected test(s). Avoid repeated full-suite runs without a reason.

### Step C — Load-only diagnostic
Use the existing `phase4_timed_run.py` wrapper. Start with the 30-minute configuration in load-only mode.

The goal is to confirm that required inputs load, the engine initializes, and transition/reliability structures build successfully while runtime and memory are measured. Do not evaluate all queries in this step.

If it fails, stop and report the exact error and missing dependency/input. Do not retry the same failing operation without a change.

### Step D — Small validation run
Only if load-only succeeds, run the 30-minute configuration with a maximum of 300 validation queries.

Verify:
- Engine initialization and query evaluation succeed.
- Candidate generation and ranking work.
- Metrics calculate successfully.
- Expected output files are created.
- Runtime and peak memory are recorded.

If it fails, diagnose the specific issue before proceeding.

### Step E — Full validation experiments
After the small run succeeds, assess feasibility using measured runtime and memory.

1. Run the complete 30-minute validation experiment.
2. Then run the complete 60-minute validation experiment if its inputs and execution path are verified.

Do not start an unlimited run before reporting the measured small-run performance and expected resource requirement. Do not run configurations simultaneously unless the implementation and available resources clearly support it. Do not use a query limit for final results. Never present partial results as complete.

If full execution is not feasible, preserve diagnostics and clearly state what remains incomplete.

## 5. Research correctness checks

Verify these before accepting results:
1. Training and validation data are separated correctly.
2. Test-period events are not used to construct validation recommendations.
3. The temporal gap is applied correctly for each configuration.
4. D0 and D1 differ only in the intended reliability-related treatment.
5. Candidate generation is consistent across compared models.
6. Source-known and source-unknown queries are classified correctly.
7. Top-K metrics use the correct denominator.
8. Bootstrap resampling follows the intended clustering method.
9. Missing values, zero denominators, and empty candidate sets are handled correctly.
10. Every reported result maps to the exact configuration that produced it.

Do not change experimental definitions to improve numerical results.

## 6. Previously verified Phase 3 aggregate findings

These are background results from saved Phase 3 aggregate metrics, **not Phase 4 results**.

| Metric | 30-minute | 60-minute |
|---|---:|---:|
| Validation queries | 15,047 | 15,613 |
| Known-source queries | 510 | 549 |
| Unknown-source queries | 14,537 | 15,064 |
| Total Top-20 hits | 2 | 2 |
| Known-source Top-20 hits | 0 | 0 |
| Unknown-source Top-20 hits | 2 | 2 |

The source-known/source-unknown split was consistent across the eight model configurations and D0/D1 treatments in the saved aggregate metrics.

**Definition:** `source_known` means the source video has at least one outgoing transition in the training data. It does not merely mean the source video exists in the training catalogue.

Do not report these Phase 3 findings as Phase 4 results or assume Phase 4 will reproduce them.

## 7. Synthetic experiment — unresolved generator details

The paper describes:
- 60 movies.
- Sequence length of 20.
- Four favored next movies per state.
- Synthetic user counts from 50 to 1,600.
- First 16 events for training.
- Event 17 as the test event.
- Laplace smoothing alpha = 0.1.

The exact favored-successor probability and non-favored sampling procedure must be verified from the original generator implementation. Locate the original code if present. Do not substitute parameters from a different synthetic experiment. If the original generator cannot be found, report these details as unresolved.

## 8. Environment and reproducibility

The paper reports Python 3.12, Windows, and 16 GB RAM. Exact library versions are not yet verified.

Inspect the actual experiment environment. If appropriate, capture installed package versions in `versions.txt` using that environment. Do not report versions from another environment as the versions used in the research.

## 9. Required outputs

Keep diagnostics separate from existing Phase 3 and Phase 4 research outputs. Use a dedicated output directory.

Produce:
1. Concise execution log.
2. Test results.
3. Runtime and peak-memory measurements.
4. Phase 4 validation result files.
5. Model-wise and treatment-wise metrics.
6. Top-10 and Top-20 results.
7. Source-known/source-unknown breakdowns, if supported by the implementation.
8. Bootstrap results and uncertainty estimates.
9. Reproducibility record with configuration and library versions.
10. `PHASE4_EXECUTION_REPORT.md`.

Clearly label each item as verified, diagnostic, partial, or unavailable. Never fabricate values, metrics, confidence intervals, or execution times.

## 10. Paper correction requirements

After experiments finish, inspect the existing paper and identify only evidence-supported corrections to:
- Experimental methodology.
- Dataset split description.
- Model and treatment descriptions.
- Evaluation metrics.
- Numerical results and tables.
- Statistical significance and confidence intervals.
- Runtime and hardware information.
- Limitations.
- Reproducibility information.

Do not rewrite the entire paper. Do not replace existing numerical values unless new execution evidence or source evidence justifies the change. Do not claim Phase 4 is complete until the required full experiment has finished and its outputs have been checked.

## 11. Working rules

- Work in the existing project folder.
- Prefer the smallest necessary code changes.
- Avoid unnecessary loops, repeated checks, duplicate experiments, and redundant diagnostics.
- Do not repeatedly execute an expensive command without a specific diagnostic reason.
- Do not launch all configurations simultaneously.
- Do not run the full experiment before load-only and small-run checks succeed.
- Do not delete or overwrite original data, source files, or existing research outputs.
- Do not install unnecessary packages.
- Do not invent missing files, values, or results.
- Do not claim success based only on code inspection.
- Inform the user before expensive computation.
- If an input is missing, request only that specific file or provide the exact local command needed to generate it. Do not request the entire raw dataset when a smaller derived file is sufficient.

## 12. Start here

First inspect the connected folder and report:
1. What is already implemented?
2. Which required files are available?
3. What is missing?
4. Which tests and diagnostics can run immediately?
5. What is the shortest safe execution plan?

Then proceed through the stages above in order.

**Final objective:** complete the remaining ReliMarkov experiment work with the minimum necessary computation while preserving scientific correctness and reproducibility. The priority is a verified research result, not unnecessary code changes or long-running experiments.
