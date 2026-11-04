# Phase 3 Candidate-Generation and Evaluation Audit

## Scope

This audit examines the Phase 3 validation runner and its saved 30-minute and 60-minute validation metric and ranking-overlap CSVs in `outputs/phase3_validation/run_20261103_213050_674910100/`. No experiment was run, no source or historical output was changed, and no test split, test outcome, or test metric was accessed.

## Executive verdict

**The current comparison is not methodologically fair as a controlled comparison of model scoring.** All variants use the same validation rows, training catalogue as their broad eligibility universe, no explicit item-exclusion filter, and the same tie-break ordering. However, M1 receives a popularity fallback of 100 items while every M4/M5 variant receives only 50. The evaluated candidate pools therefore differ systematically. This confounds ranking-model comparisons with candidate-generation policy, even though M1/M4-W4 happen to have identical Top-K sets in these saved results.

The reported Top-K overlap is a per-query **set** overlap, not a rank-position comparison. The metric denominators are consistent across models for the same subset, but the all-query denominator includes validation targets outside the training catalogue and therefore impossible for any current recommender to return. Finally, the coverage values repeated on subset rows are calculated over all validation queries, not the named subset.

Before the next experiment, standardize the candidate-generation policy across models and clarify evaluation denominator/coverage semantics. Then rerun validation only and report candidate-eligible target counts alongside overall and reachable-target results. The proposed steps are listed below; no correction has been implemented.

## 1. Candidate-generation trace

The Phase 3 engine builds `global_top_100` by descending training popularity, breaking popularity ties by ascending video ID; `global_top_50` is the first 50 entries of that same list ([`kr_experiments_phase3.py`](./kr_experiments_phase3.py#L188-L189)).

For each query source `s`:

- **M1:** `set(Counter(s)) ∪ global_top_100`. It scores each candidate by the integer transition count `C(s,v)` only ([candidate branch](./kr_experiments_phase3.py#L414-L423)).
- **M4-W1:** `set(Counter(s)) ∪ global_top_50`. Its score is `C(s,v) + m_v` with `alpha=1`; query reliability is passed into the helper but not used for W1 ([M4 branch](./kr_experiments_phase3.py#L425-L438), [score helper](./kr_experiments_phase3.py#L117-L126)).
- **M4-W2/W3/W4:** the source's weighted transition table keys unioned with `global_top_50`; scored using the selected weighted evidence and the same W1 prior mass convention.
- **M5-A/B/C:** W4 transition keys unioned with `global_top_50`; each variant differs in its score calculation, not in this candidate-source rule ([M5 branch](./kr_experiments_phase3.py#L440-L466)).

The weighted transition contributions use reliability values clipped to a positive minimum, so observed source-transition edges have positive W2/W3/W4 accumulation. In effect, the M4/M5 local candidate support is the source's training transition-neighbor set, like M1's, but their popularity backfill is smaller.

The returned candidates are ranked by score descending, then raw training popularity descending, then video ID ascending ([`_rank`](./kr_experiments_phase3.py#L402-L412)). All model branches use this ranker. Each model returns the first 20 ranked items.

### Why the pool sizes differ

The difference is explicitly caused by using `global_top_100` for M1 and `global_top_50` for every M4/M5. Since top-50 is contained in top-100, the per-query sets are:

`M1 = local_neighbors ∪ top100`

`M4/M5 = local_neighbors ∪ top50`

Thus M1's pool is a superset or equal for each query, and models do not have the same eligible candidate set. The small excess over the fallback size comes from local neighbors outside the fallback list.

The CSVs confirm the expected means and show the median and 95th percentile at the fallback sizes:

| Gap | Model | Mean pool | Median | 95th percentile |
|---|---|---:|---:|---:|
| 30 min | M1 | 100.040739 | 100 | 100 |
| 30 min | each M4/M5 | 50.040805 | 50 | 50 |
| 60 min | M1 | 100.042913 | 100 | 100 |
| 60 min | each M4/M5 | 50.042977 | 50 | 50 |

References: [`validation_metrics_30min.csv`](./outputs/phase3_validation/run_20261103_213050_674910100/validation_metrics_30min.csv), [`validation_metrics_60min.csv`](./outputs/phase3_validation/run_20261103_213050_674910100/validation_metrics_60min.csv).

## 2. Candidate eligibility, fallback, filtering, and ties

- **Broad eligible universe:** the engine loads the training catalogue, and it asserts that every returned recommendation is in that catalogue ([validation assertions](./kr_experiments_phase3.py#L556-L589)). This does not make the *per-query candidate pools* identical: each is a smaller, model-dependent subset of the training catalogue.
- **Popularity fallback:** not the same size. M1 gets top-100; all M4/M5 variants get top-50. The lists share the same popularity ordering, with top-50 nested in top-100.
- **Local transition candidates:** same edge support in practice: integer-count keys for M1/W1 and positive-weight transition keys for the weighted variants. The local support and popularity backfill are unioned; duplicates are naturally removed by `set`.
- **Filtering:** there is no explicit per-query exclusion of previously seen items, the source item itself, or other unavailable items in these candidate branches. The policy is consistently “no such filtering,” rather than a model-specific filter. Whether self/seen-item filtering is appropriate is an unresolved task-definition choice; it should be specified before the next experiment and applied uniformly if required.
- **Tie-breaking:** same deterministic secondary order across models: training popularity descending, then video ID ascending. Primary scores differ by model. M1 has no popularity prior in its score; M4/M5 scoring includes their respective prior terms.
- **Candidate-pool assertion:** the implementation checks the returned Top-20 IDs against the training catalogue, but does not separately assert that every member of the full pre-truncation candidate set is in the catalogue. The constructed sources are training transition keys and training popularity keys, so a separate invariant check would be useful but this audit found no evidence in the saved outputs of an out-of-catalogue recommendation.

## 3. Overlap and query alignment

All model predictions are generated by iterating the same `engine.val_df` rows in the same order for each treatment/model ([validation prediction loop](./kr_experiments_phase3.py#L556-L589)). Overlap is then computed by zipping those prediction lists, so the saved comparisons are over corresponding validation rows under the current implementation ([overlap calculation](./kr_experiments_phase3.py#L592-L642)). There is no explicit query-ID alignment assertion in `_overlap_rows`; the current alignment follows from the shared loop/order rather than being verified at comparison time.

The helper converts the first K recommendations to sets and computes intersection divided by K, plus Jaccard ([`topk_overlap`](./kr_experiments_phase3.py#L144-L151)). Upstream, every model/query must return exactly 20 recommendations or the run raises an assertion ([validation loop](./kr_experiments_phase3.py#L566-L570)); candidate pools are always at least 50/100 here. So there are no short lists or missing recommendation slots being padded, dropped, or treated inconsistently in these artifacts.

This overlap is **not position-sensitive**. A list with the same Top-10 items in a different order still scores 100% overlap. Therefore:

- M1 vs M4-W4 has overlap 1.0 at both K values for D0 and D1 at both gaps: the sets match, not necessarily that every item's rank is unchanged.
- M1 vs M5-B has the following D0 Top-10/Top-20 overlaps: 0.999900/0.999960 at 30 minutes and 0.999853/0.999942 at 60 minutes. For D1 they are 0.999914/0.999970 and 0.999891/0.999949, respectively. These near-1 set overlaps similarly say little about rank movement within the set.
- The overlap CSV cannot show differences beyond Top-20 and does not by itself establish equivalence of scoring rules.

Saved overlap files: [`ranking_overlap_30min.csv`](./outputs/phase3_validation/run_20261103_213050_674910100/ranking_overlap_30min.csv), [`ranking_overlap_60min.csv`](./outputs/phase3_validation/run_20261103_213050_674910100/ranking_overlap_60min.csv).

## 4. Metrics, denominator, and observed hits

`_case_metrics` scores the target's position in the returned list and returns zero for Hit/NDCG/MRR when the target is absent from Top-K ([metric calculation](./kr_experiments_phase3.py#L473-L482)). The all-query metric denominator is the number of validation transition rows. Every model/treatment has 15,047 rows for the 30-minute gap and 15,613 for the 60-minute gap. The query set and overall denominator are therefore consistent across models for each gap.

The all-query denominator includes target videos not in the training catalogue. Such targets cannot be returned because all recommendations are constrained to that catalogue. In the saved artifacts:

- 30 min: 2,936 of 15,047 targets are in the training catalogue; 12,111 are not.
- 60 min: 3,028 of 15,613 targets are in the training catalogue; 12,585 are not.

The `target_known` subset means target is in the training catalogue; it does **not** mean the target was in that query's generated candidate pool. The source subsets are defined by whether the source has any outgoing training transition count; `source_unknown` is not a claim that the source video itself is absent from the catalogue ([subset flags](./kr_experiments_phase3.py#L575-L583)). These subsets overlap and are not a partition.

The CSV confirms exactly **two Top-20 hits** for every model and treatment at each gap. Both hits are in `target_known`. This is consistent with the hit rates (`2 / 15,047` and `2 / 15,613`). The all-subset aggregate Hit@10/20, NDCG@10/20, and MRR@10/20 values are exactly the same across all eight model variants and both D0/D1 treatments within each gap (zero range at the saved CSV precision). This is a highly sparse validation signal; it cannot support a claim that one model improves performance.

Including catalogue-unknown targets in the all-query denominator is a valid end-to-end convention if that is the declared task, but it mixes ranking performance with catalogue coverage. For an interpretable next-item comparison, retain the all-query metrics and also clearly report metrics conditional on a stated reachable-target set; do not silently discard unknown targets.

### Coverage scope inconsistency

`run_gap` passes the full per-model `records` list as `all_case_records` even when `selected` is a subset ([aggregation call](./kr_experiments_phase3.py#L652-L666)). `_aggregate_metrics` computes `unique_recommendations` and `recommendation_coverage` from `all_case_records`, while `n_cases`, ranking metrics, pool sizes, and author diversity use the selected subset ([aggregation](./kr_experiments_phase3.py#L516-L555)). Consequently, the coverage and unique-recommendation columns repeated on `target_known`, `source_known`, and `source_unknown` rows are full-validation-set coverage, not coverage for the named subset. Overall rows are unaffected. Either compute coverage from the selected rows or label/document the columns as full-validation coverage; fix the ambiguity before interpreting subgroup coverage.

## 5. Confirmed issues and unresolved choices

### Confirmed implementation/methodology issues

1. **Model-dependent candidate-pool size:** M1 gets top-100 backfill; M4/M5 get top-50. This is the principal fairness problem and confounds scoring with candidate generation.
2. **Subset coverage scope mismatch:** coverage and unique recommendation counts are calculated over all queries even on subset rows.
3. **Overlap diagnostics conceal order changes:** the saved overlap is set overlap only, not a rank-sensitive comparison. This is a diagnostic limitation rather than an incorrect set-overlap calculation.
4. **No runtime query-key check in overlap:** current query alignment is established by shared iteration, but the overlap function zips without validating query identities or lengths. This is safe for the current construction but fragile to future code changes.
5. **Sparse outcome signal:** only two Top-20 hits per gap, with all variants sharing the same aggregate metrics. Given the present candidate sets and denominator, the evaluation does not meaningfully distinguish the models.

### Decisions to make explicitly

- Whether the intended evaluation is all validation targets or only targets reachable within the training catalogue/candidate set. Best practice here is to retain and label all-query metrics and report reachable-target metrics separately, with denominators.
- Whether a recommendation should exclude the source item or any previously consumed items. The current implementation does not do so.
- Whether the next experiment intends to compare **ranking given a candidate set** or **end-to-end retrieval plus ranking**. The candidate universe must be standardized for the former; for the latter, candidate-generation differences must be treated as part of the method and separately diagnosed.

## 6. Proposed correction plan for the next experiment (not implemented)

1. **Freeze one candidate-generation policy for every model.** Use the same query-specific union for M1/M4/M5: the same source-neighbor support and one shared popularity fallback size/list. Choose the fallback size as an explicit protocol decision before running; do not pick it based on validation scores. Assert candidate-set equality across models per query where the comparison is intended to isolate ranking.
2. **Specify and apply filtering uniformly.** Decide whether to remove the source and/or seen items. Apply the same rule to every model and record candidate counts before and after filtering.
3. **Strengthen alignment checks and diagnostics.** Carry validation query IDs alongside each prediction and assert identical IDs/order and exactly K recommendations before comparing. Keep set overlap, but add rank-sensitive diagnostics (for example, per-position agreement or a rank correlation on common items).
4. **Clarify evaluation populations.** Report all-query and catalogue-known-target results with explicit denominators, plus candidate-reachable-target counts/results under the common candidate policy. Keep known-source/unknown-source subsets and label them by their exact edge-support definition.
5. **Correct coverage aggregation.** Either calculate unique recommendations and coverage from each selected subset, or rename the columns to explicitly state that they are full-validation coverage and omit duplicate subgroup values.
6. **Run a new validation-only experiment into a fresh versioned output directory.** Preserve this audit's source artifacts and all prior outputs. Do not execute or inspect frozen-test evaluation.

**Final verdict:** the current comparison uses the same queries and broad training-catalogue boundary, but not the same per-query candidate universe; therefore it is not a fair isolated comparison of M1, M4, and M5 ranking behavior. Before the next experiment, standardize candidate generation and repair/clarify subgroup coverage and evaluation-population reporting. No changes were made to the implementation or historical results during this audit.
