# Phase 4 Methodology: Controlled Candidate-Generation Validation

## Purpose and scope

Phase 4 compares M1, M4-W1/W2/W3/W4, and M5-A/B/C while holding each validation query's candidate set constant across models. It runs only for the existing 30-minute and 60-minute train/validation splits and uses the approved D0 and D1 reliability treatments. It does not read test split files, evaluate test outcomes, or alter any split. Phase 3 scripts and outputs remain unchanged.

The runner is `kr_experiments_phase4.py`; unit tests are in `test_phase4_validation.py`. Each invocation writes to a new timestamped directory under `outputs/phase4_validation/` and refuses to overwrite an existing run directory.

## Candidate policy

For source video `s`, define:

- `N_train(s)`: all distinct destination videos observed after `s` in training transitions.
- `P100_train`: the global Top-100 videos ordered by training popularity descending, then video ID ascending.

The single per-query candidate set is:

`C(s) = N_train(s) ∪ P100_train`.

The same immutable candidate set is constructed once per validation query and passed to each model under both D0 and D1. The runner constructs a model-to-candidate-set mapping and asserts exact set equality for every model/query/treatment. It also checks that the set is within the training catalogue and that each returned recommendation is in that catalogue. Candidate construction is based only on training transitions and popularity.

The controlled experiment retains the existing no-exclusion policy: it does not remove the source video or previously seen videos. This policy is the same for every model. No pre-Top-20 candidate filtering is performed.

Ranking is deterministic for all models: score descending, training popularity descending, then video ID ascending.

## Scoring and reliability

Phase 4 reuses the established Phase 3 reliability and model-scoring helpers without changing them. For event `e`:

`r(e) = clip(watch_ratio + 0.3*is_like + 0.2*is_forward - 0.5*is_hate, 0.1, 1.5)`.

For positive durations, `watch_ratio = clip(play_time_ms / duration_ms, 0, 1)`. For nonpositive durations, D0 uses `0.0` and D1 uses `0.9016`.

With integer training transition count `C(s,v)`, reliability-weighted counts `W2/W3/W4`, query reliability `r_q`, training popularity prior mass `m_v`, treatment-specific training mean W4 `mu`, and standard integer-count EB concentration `alpha_EB`, scores are:

- M1: `C(s,v)`.
- M4-W1: `C(s,v) + m_v` (`alpha=1`; independent of query reliability).
- M4-W2/W3/W4: `r_q * Wj(s,v) + m_v`.
- M5-A: `r_q * W4(s,v) + alpha_EB * m_v`.
- M5-B: `r_q * (W4(s,v) / mu) + alpha_EB * m_v`.
- M5-C: `r_q * W4(s,v) + (mu * alpha_EB) * m_v`.

The standard EB concentration remains estimated on integer counts and is loaded separately from the reliability-weighted M4/M5 statistics. Phase 4 does not implement M5-Ext or fit a weighted-likelihood EB model.

## Validation data access

The runner reads only the existing training and validation transition files, plus training catalogue/popularity/author metadata and the standard integer-count EB parameter file. To retrieve event reliability inputs without parsing frozen-test outcomes, it first scans only `time_ms` from the standard interaction CSVs and creates a row mask for timestamps within the approved training-through-validation window. It then asks the CSV parser to skip all other rows before parsing event columns. A parsed-row count is checked against the timestamp mask. The split's test Parquet files are never opened.

The training window and validation cutoff are fixed constants copied from the Phase 3 validation implementation; no boundary or configuration is changed. The 30-minute and 60-minute validation rows are loaded explicitly by filename.

## Query alignment and ranking diagnostics

Each query gets a stable ID from its existing split row order and gap. Before overlap is calculated, left and right query-ID sequences must match exactly; reordered or missing queries raise an assertion.

The overlap output retains per-query mean Top-K set overlap and Jaccard and adds:

- **Position agreement@K:** fraction of positions whose video IDs match exactly.
- **Mean absolute rank displacement:** average absolute rank difference for items appearing in both Top-K lists.

These are diagnostics, not performance metrics. They distinguish same-item-set results from identical rank order.

## Evaluation populations and denominators

Video recommendation metrics are reported separately for:

1. `all_validation_targets`: all validation query targets.
2. `target_videos_in_training_catalogue`: targets available in the training catalogue.
3. `target_videos_in_query_candidate_set`: targets included in that query's common candidate set.

Next-author metrics are separate from next-video metrics. The author ranking is formed by traversing the full ranked video candidate list and retaining the first occurrence of each mapped author, skipping videos without an author mapping. The three author populations are:

1. `all_validation_targets`: all validation rows, with unknown target authors counted as unscorable and excluded from the metric denominator.
2. `target_authors_in_training_catalogue`: target authors represented by at least one training-catalogue video.
3. `target_authors_in_query_candidate_authors`: target authors represented among authors of that query's common video candidate set.

Every output row contains `n_population_queries`, metric denominator `n_cases`, `n_unscorable`, and integer `hit_count@10` / `hit_count@20`. Hit rate, NDCG, and MRR use `n_cases` as their denominator. User-cluster bootstrap 95% intervals are also reported for Hit/NDCG/MRR. Video and author coverage are calculated only over the selected population's scorable recommendations; they are not silently copied from the full-validation set.

Recommendation coverage is the count of distinct recommended video IDs divided by training-catalogue video count for next-video rows. For next-author rows, it is distinct recommended authors divided by the set of authors represented in the training catalogue.

## Limitations

- Candidate generation is intentionally controlled and therefore does not compare the historical Phase 3 M1 Top-100 vs M4/M5 Top-50 retrieval policies.
- No source/seen-item exclusion is applied; changing this policy would define a different experiment.
- Next-author results depend on the completeness of the training author map. Rows without target-author metadata are counted as unscorable for author metrics.
- Validation metrics can remain sparse; candidate equality improves experimental control but does not guarantee sufficient statistical power or establish model superiority.
