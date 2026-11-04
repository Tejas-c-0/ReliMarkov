# Mathematical and Implementation Consistency Audit

**Project:** Reliability-Aware Markov Chain Recommendation for Personalised Content  
**Date:** 2026-11-03  
**Scope:** Static inspection of implementation and existing validation artifacts only.

No existing files were modified, deleted, renamed, or overwritten. No test evaluation was run, no test outcomes or metrics were inspected, and no new experimental results were generated. Dataset splits and configurations were not changed.

## 1. Verified implementation behavior

### Reliability weights and training aggregates

In [`kr_experiments.py`](./kr_experiments.py):

- `ExperimentEngine.__init__` loads the EB JSON parameters at lines 131–137; video and author `alpha0` values are assigned at lines 134–135.
- Lines 148–159 filter events to `tab == 1` and `is_click == 1`, calculate clipped watch ratio, and set event reliability to
  `r = clip(watch_ratio + 0.3*is_like + 0.2*is_forward - 0.5*is_hate, 0.1, 1.5)`.
  Zero-duration watch ratio is set to `0.9016` at lines 150–153.
- Training transition tables are built at lines 169–195. For transition `i`, source and target reliabilities are looked up at lines 179–180, each with fallback `0.68`; W4 is `sqrt(r_src * r_tgt)` at line 181. Integer counts and weighted W4 counts are accumulated at lines 185–192. The W4 mean at line 194 is the arithmetic mean across the training transition rows (not unique source-target pairs).
- `get_query_weight` at lines 197–198 returns source-event reliability with the same `0.68` fallback.

For source state `s` and target `v`, let `C_sv` be the integer transition count, and let:

- `S2_sv = sum_i r_src,i` for transitions `s -> v` (W2),
- `S3_sv = sum_i r_tgt,i` (W3),
- `S4_sv = sum_i sqrt(r_src,i * r_tgt,i)` (W4).

The current scoring functions are:

- **M4 W1/W2/W3/W4:** `rec_m4_rw`, [`kr_experiments.py:237–253`](./kr_experiments.py). W1 selects `C`; W2 selects `S2`; W3 selects `S3`; W4 selects `S4` (lines 239–247). All four then score `v` as `r_q * table[s,v] + α*m_v`, where `r_q` is source-query reliability and the validation runner uses `α=1` (lines 248–250 and 333–336).
- **M5-A:** `rec_m5a_raw`, [`kr_experiments.py:255–263`](./kr_experiments.py), scores `r_q*S4_sv + α_EB*m_v`.
- **M5-B:** `rec_m5b_norm`, [`kr_experiments.py:265–274`](./kr_experiments.py), sets `μ=mean_w4_train`, `r_q_norm=r_q/μ` and scores `(r_q/μ)*(S4_sv/μ) + α_EB*m_v`.
- **M5-C:** `rec_m5c_calib`, [`kr_experiments.py:276–285`](./kr_experiments.py), sets `α_eff=λ*α_EB` and scores `r_q*S4_sv + λ*α_EB*m_v`.

The candidate sets in these functions are each observed outgoing targets for `s` union the top 50 globally popular training videos. Each function returns the top 20 by score. The common candidate construction and ranking are in the cited function ranges.

### EB estimator and parameter use

[`fit_eb_concentration`](./estimate_eb.py#L21) groups integer training transitions by source-target pair (lines 24–28), obtains target prior mass from the training popularity base measure (lines 29–30), and optimizes the Dirichlet-multinomial log evidence over `[0.1, 50000]` (lines 32–45). The evidence terms are:

\[
\ell(\alpha)=\sum_s[\log\Gamma(\alpha)-\log\Gamma(N_s+\alpha)]
+\sum_{(s,v):C_{sv}>0}[\log\Gamma(C_{sv}+\alpha m_v)-\log\Gamma(\alpha m_v)].
\]

The omitted multinomial coefficient is independent of `α` and therefore does not change this one-dimensional optimizer. The script runs this for video and author transitions and writes JSON at [`estimate_eb.py:68–96`](./estimate_eb.py#L68). `ExperimentEngine` reads the corresponding per-gap JSON at [`kr_experiments.py:131–137`](./kr_experiments.py#L131).

### Other reliability definitions found

- [`reliability_diagnostics.py:12–19`](./reliability_diagnostics.py#L12) imputes zero-duration watch ratio as `0.5`, then uses the same displayed feedback coefficients and `[0.1, 1.5]` clipping as the experiment. It inspects only `tr.head(20000)` at line 34; its joins at lines 35–40 have no missing-weight fill, and it computes W1–W4 at lines 42–45.
- [`22_chrono_split.py:65–89`](./22_chrono_split.py#L65) uses zero watch ratio for nonpositive duration, includes `0.3*long_view` in addition to the feedback terms, and clips event weights to `[0.05, 1.5]`. This is a separate historical implementation, not the current `ExperimentEngine` formula.

The stored EB values and mean W4 figures are reported in the preceding `COPILOT_READ_ONLY_AUDIT.md`; they were not recalculated here.

## 2. Mathematical consistency findings

### 2.1 M4-W1 is not an unweighted recommendation score

**Confirmed behavior; a bug if “unweighted” means no reliability enters scoring.** W1 selects integer counts, but the shared M4 formula still multiplies those counts by query reliability `r_q` (`kr_experiments.py:237–250`). With a nonconstant popularity prior, multiplying only the data term can change the balance between transition evidence and prior:

\[
score_{W1}(v)=r_q C_{sv}+\alpha m_v
\quad\ne\quad C_{sv}+\alpha m_v.
\]

`r_q` is constant over candidates in a given query, so it does not change ordering among observed targets when the prior is absent or identical. But it does change their ordering relative to prior-scored candidates, and can affect ranking when `m_v` differs. Therefore W1 is an **unweighted historical transition-count ablation with query-time reliability weighting**, not a fully unweighted baseline. If the intended ablation removes reliability entirely, the corrected equation is `C_sv + α m_v` (or use M1's empirical-count ranking with its documented fallback). Which interpretation is desired is a terminology/method decision; the current behavior itself is verified.

### 2.2 M5-B applies two divisions by μ

**Confirmed implementation; a bug relative to “normalize W4 to mean 1” if that refers only to transition W4.** Since `S4_sv` is a sum of per-transition W4 values, dividing by `μ` normalizes those historical weights. The code also divides `r_q` by the same `μ`, so the data term is:

\[
\frac{r_q}{\mu}\frac{S4_{sv}}{\mu}
=\frac{r_q S4_{sv}}{\mu^2}.
\]

If the intended operation is to make the *training per-transition W4* mean one, define `q_i=W4_i/μ`; then the historical pseudo-count is `S4'_sv=S4_sv/μ`. Keeping query reliability as currently defined yields the score:

\[
score(v)=r_q\frac{S4_{sv}}{\mu}+\alpha_{EB}m_v.
\]

This is the direct normalization of W4 only. Dividing query reliability too would require an additional reason and a separately justified scale. W4 is a geometric mean of two event reliabilities, whereas `r_q` is one event reliability; they need not have the same mean. If the aim is to normalize both populations, estimate and state their separate means (`μ_W4` and `μ_query`) and use the explicit factor `r_q/μ_query * S4/μ_W4`; this is a different model.

### 2.3 M5-C calibration

**The parameter has a defensible scale-equivalence derivation, but its provenance/label is unresolved.** If the intended normalized historical evidence is `r_q*S4/μ` and the prior is left at `α_EB*m_v`, multiplying the *entire ranking score* by positive `μ` preserves the ranking and gives:

\[
\mu\left(r_q\frac{S4_{sv}}{\mu}+\alpha_{EB}m_v\right)
=r_q S4_{sv}+(\mu\alpha_{EB})m_v.
\]

Thus `λ=μ` in M5-C is mathematically equivalent, for ranking, to normalizing historical W4 by μ while preserving the original prior scale. This derivation assumes the query reliability is not also divided by μ and that multiplying weighted evidence by one global scale is the intended modeling choice. It does **not** make M5-C a weighted-data EB fit or establish that the unweighted EB value is statistically calibrated for weighted counts.

The runner passes `engine.mean_w4_train` at [`kr_experiments.py:339`](./kr_experiments.py#L339), while the output label is `M5C_ra_ebmm_calib_0.55`; the function default is `0.5543` at line 276. The displayed label is therefore not the actual per-gap parameter. Treat the derivation as a principled scale conversion, not empirical calibration, unless separate validation-based calibration is defined and reported.

### 2.4 Integer-count EB reused with weighted pseudo-counts

**Statistical coherence is not established; not automatically a coding error.** The estimator maximizes evidence for integer counts under a categorical/Dirichlet-multinomial model. M5 instead inserts reliability-weighted fractional totals and query reliability into a score while retaining that integer-count estimate of `α`.

Dirichlet posterior updates accept nonnegative fractional exponents as a generalized/power likelihood:

\[
p(\theta\mid data)\propto Dir(\theta;\alpha m)\prod_i p_{s_i,t_i}(\theta)^{q_i}.
\]

Its update has parameters `αm_v + Σ_i q_i 1[t_i=v]`. But arbitrary fractional/purpose-built reliability weights do not by themselves define ordinary categorical sampling counts. Substituting fractional values into the gamma-function Dirichlet-multinomial expression is generally a **pseudo-evidence/generalized Bayes** construction, not automatically a normalized sampling marginal likelihood. Also, the current score applies `r_q` again to historical weighted evidence, which is an additional model choice.

Two defensible paths, with different assumptions:

1. **Keep standard EB:** use integer counts for M1–M3 and explicitly treat M4/M5 weights as a separate ranking heuristic. For M5, choose and state a scale convention (e.g. normalize historical `S4` by μ and retain α on the integer-count scale). Do not describe the resulting α as EB-estimated for the weighted model.
2. **Define a weighted EB extension:** specify the weighted/power likelihood and how reliability weights are obtained, then estimate α under that declared objective. A gamma-function expression using `C^w_sv=Σ_i q_i 1[t_i=v]` and `N^w_s=Σ_v C^w_sv` is a possible generalized objective, but fractional counts make it pseudo-evidence unless a proper observation/reliability model justifies it. A generative latent-reliability model would provide a stronger probabilistic basis but requires assumptions and implementation beyond the current code.

Mean normalization aligns average pseudo-count scale but does not, by itself, prove correct uncertainty, independence, or likelihood assumptions.

### 2.5 Zero-duration and missing-weight handling

**Cross-script inconsistency is confirmed.** The primary experiment imputes watch ratio `0.9016`; the diagnostic script uses `0.5`; the historical split script uses `0` for nonpositive duration and also differs in feedback terms and minimum clipping. The diagnostic left joins can leave weights null, while the experiment uses `0.68` for any missing source/target lookup and for missing query reliability. Thus the reliability diagnostic is not an exact reproduction of the primary experiment.

Whether to impute, exclude, or model nonpositive duration as missing is unresolved. If retaining imputation, use one documented rule and compute the imputation on training data only; if missing event matches are possible, either fail/report their count or use a justified fallback consistently rather than silently masking merge failures. The static audit did not establish how often missing matches occur.

## 3. Experiment identity: M5-Ext

No model called **M5-Ext** is implemented in the inspected model list or methods. The validation list explicitly includes M5-A, M5-B, and M5-C at [`kr_experiments.py:327–340`](./kr_experiments.py#L327).

M5-C is a calibrated/scaled score using a previously estimated integer-count EB α. A weighted-likelihood Empirical Bayes extension would instead define weighted sufficient statistics and an explicit hyperparameter objective (or a reliability-generative model), then estimate α for that weighted model. It is not equivalent to multiplying the old α by λ. Keep M5-C and any future M5-Ext separately named and describe their assumptions independently.

## 4. Ranking diagnostics and artifact sufficiency

`compute_case_metrics` at [`kr_experiments.py:47–71`](./kr_experiments.py#L47) computes target rank, Hit@10/20, NDCG@10, MRR@10, and author hit metrics. `run_all_validation` at [`kr_experiments.py:317–404`](./kr_experiments.py#L317) stores aggregate metrics, target-known subset metrics, `unique_recs`, and `cand_coverage`.

Existing validation artifacts therefore provide:

- **MRR and NDCG:** aggregate all-query and target-known values are present. They reflect the target's position, not all pairwise/item ranking changes.
- **Recommendation coverage/diversity proxy:** `unique_recs` and `cand_coverage` measure the total number/fraction of distinct videos recommended across all validation queries. They are not per-query candidate-set size, intra-list diversity, or author/category diversity.
- **Target-known performance:** metrics are available for targets present in the training catalogue. This is not a known-source subset.

They do **not** retain per-query ranked lists or scores, so they cannot establish top-K overlap between M1/M4/M5, quantify item rank shifts, compute rank correlation, or calculate per-query candidate-set size. They also do not report metrics specifically restricted to source states observed in training. In the stored validation CSVs, M1, M4, and M5 have identical displayed target metrics for each gap while `unique_recs` differs for some M5 variants. Equal aggregate target metrics do not prove equal rankings: non-target ordering can change without affecting these metrics.

Useful future validation-only diagnostics (not run for this audit):

- Per-query Top-K overlap, e.g. `|TopK_A ∩ TopK_B|/K` for M1 vs M4 and M1 vs M5, plus Jaccard overlap.
- Per-query candidate-pool size **before truncating to 20**, and number of observed transitions for the source.
- MRR/NDCG and Hit@K on all cases, target-known cases, and a separately reported source-known subset; retain user-cluster uncertainty intervals.
- Recommendation coverage (`unique_recs/catalogue_size`) plus intra-list diversity and author-level diversity, if relevant to the research question.
- Store or summarize per-query ranks/scores in validation outputs so ranking changes are measurable. Preserve existing historical results; any new artifacts should use new filenames after approval.

## 5. Confirmed bugs vs. methodological decisions

### Confirmed implementation discrepancies

1. M4-W1 is labeled unweighted but still uses query reliability in its data score.
2. M5-B divides both `r_q` and `S4` by μ, giving `1/μ²` scaling.
3. M5-C's result label does not match the μ value passed by the runner.
4. Reliability preprocessing differs between the main experiment, diagnostic, and older split script; the diagnostic also does not fill unmatched reliability joins.
5. Existing validation artifacts do not contain per-query ranking/candidate diagnostics needed to compare recommendation lists directly.

Items 1–3 are bugs **if** their names/comments express the stated normalization/ablation intentions. Their intended semantics should be confirmed before changing implementation.

### Methodological decisions, not established bugs

- Whether EB should apply to weighted pseudo-counts, and what objective makes that estimator meaningful.
- Whether W4 normalization should also normalize query reliability (and, if so, by which population mean).
- Whether zero/invalid duration should be imputed, excluded, or treated by a measurement model.
- Whether λ=μ is the desired scale-equivalence convention, or λ should instead be independently calibrated on validation under a declared tuning protocol.
- Whether candidate ranking should include only transition targets plus top-50 popularity candidates or a larger training catalogue.

## 6. Recommended corrected equations

Subject to confirming intent, a consistent reliability-weighted score family can be stated as:

\[
score_{M4,j}(v)=
\begin{cases}
C_{sv}+\alpha m_v & j=W1\text{ (fully unweighted)},\\
r_q S2_{sv}+\alpha m_v & j=W2,\\
r_q S3_{sv}+\alpha m_v & j=W3,\\
r_q S4_{sv}+\alpha m_v & j=W4.
\end{cases}
\]

If W1 is intended to weight the query but not historical transitions, name it accordingly and retain `r_q C_sv + αm_v`; it is not a fully unweighted baseline.

For mean-one normalization of historical W4 alone:

\[
S4'_{sv}=S4_{sv}/\mu,\qquad
score_{M5,norm}(v)=r_q S4'_{sv}+\alpha m_v.
\]

The raw-score form with identical ranking is:

\[
score_{M5,scale}(v)=r_q S4_{sv}+(\mu\alpha)m_v.
\]

The latter is M5-C with `λ=μ`, under the assumptions stated above. A weighted EB extension should not be assigned this name; it needs its own declared weighted likelihood/objective and concentration estimate.

## 7. Prioritized correction plan (approval required before source changes)

1. **Freeze interpretation and labels:** decide what W1 means, confirm whether M5-B normalizes historical W4 only, and state whether M5-C uses λ=μ as scale conversion or a tuned calibration. Keep M5-Ext as a distinct name unless its own implementation/specification is approved.
2. **Resolve EB scope:** choose between plug-in integer-count EB plus weighted heuristic versus a separately derived weighted/generalized EB estimator. Document assumptions before coding.
3. **Unify reliability preprocessing:** choose duration handling, clipping, signals, and missing-join policy; ensure diagnostics reproduce the primary definition or are explicitly labeled as alternate analyses.
4. **Update validation diagnostics only after approval:** add per-query Top-K overlap, candidate-pool sizes, and source-known subset reporting to new outputs, preserving all historical CSVs.
5. **Validate the approved changes on validation only:** do not access the frozen test outcomes.

**End of audit. No source changes or experiments were made.**
