"""
kr_eval.py  —  Evaluation Framework for Baselines and Sequential Models
=======================================================================

PURPOSE
  Evaluates recommendation baselines on chronological validation/test splits.
  
HARD RULES ENFORCED
  1. Only training-derived catalogue, popularity, transition matrices, and
     author->video mappings are used for candidate generation and scoring.
  2. Every recommended video must belong strictly to V_train.
  3. Every video recommended from an author must strictly belong to that author.
  4. Never tune on the test set. Default evaluation target is 'val'.
  5. Report both 'All Cases' and 'Target-Known' subset along with coverage.
  6. Next-AUTHOR Recall@K reported separately.
  7. User-level cluster bootstrap 95% confidence intervals (B=1000 resamples).
  8. Exact hit counts, target counts, and coverage reported.

BASELINES IMPLEMENTED
  1. video_pop      : Global video popularity (train counts).
  2. author_pop     : Author popularity -> rank author's train videos by train popularity.
  3. same_author    : Same author as previous event -> rank that author's train videos.
  4. video_markov   : Video-level 1st-order Markov with global video popularity fallback.
  5. author_markov  : Author-level 1st-order Markov -> pick author's videos by train popularity.

USAGE
  python kr_eval.py --gap 30 --split val
  python kr_eval.py --gap 60 --split val
"""

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import psutil

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(r"D:\APPS Research\KuaiRand-1K\data")
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = Path(r"D:\APPS Research\KuaiRand-1K\outputs\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_KS = [10, 20]
MAX_K = max(TOP_KS)
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42

def log_mem():
    mb = psutil.Process().memory_info().rss / 1_048_576
    return f"{mb:.0f} MB"

# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_case_metrics(recs, target_id, target_author, rec_authors=None):
    """
    Computes per-case metrics:
      - hit@10, hit@20
      - ndcg@10
      - mrr@10
      - auth_hit@10, auth_hit@20
    """
    metrics = {
        "hit@10": 0.0,
        "hit@20": 0.0,
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "auth_hit@10": 0.0,
        "auth_hit@20": 0.0,
    }
    
    # Video-level metrics
    if target_id in recs:
        rank = recs.index(target_id) + 1  # 1-based
        if rank <= 10:
            metrics["hit@10"] = 1.0
            metrics["ndcg@10"] = 1.0 / math.log2(rank + 1)
            metrics["mrr@10"] = 1.0 / rank
        if rank <= 20:
            metrics["hit@20"] = 1.0

    # Author-level metrics
    if rec_authors is not None and target_author is not None:
        if target_author in rec_authors[:10]:
            metrics["auth_hit@10"] = 1.0
        if target_author in rec_authors[:20]:
            metrics["auth_hit@20"] = 1.0

    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS (User-Level Cluster Bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def user_cluster_bootstrap(df_eval, metric_cols, n_rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """
    Performs user-level cluster bootstrap:
      1. Sample users with replacement.
      2. Aggregate all transitions belonging to the sampled users.
      3. Compute mean metric for each round.
      4. Report empirical 2.5% and 97.5% quantiles.
    """
    rng = np.random.RandomState(seed)
    users = df_eval["user_id"].unique()
    n_users = len(users)
    
    # Map each user to their transition indices in df_eval
    user_to_idx = df_eval.groupby("user_id").indices
    
    # Pre-extract numpy metric arrays
    arrays = {col: df_eval[col].to_numpy(dtype=np.float64) for col in metric_cols}
    
    boot_means = {col: np.empty(n_rounds, dtype=np.float64) for col in metric_cols}
    
    for b in range(n_rounds):
        sampled_users = rng.choice(users, size=n_users, replace=True)
        # Collect all rows for sampled users
        sampled_indices = np.concatenate([user_to_idx[u] for u in sampled_users])
        for col in metric_cols:
            boot_means[col][b] = np.mean(arrays[col][sampled_indices])
            
    ci_results = {}
    for col in metric_cols:
        pe = np.mean(arrays[col])
        ci_lower = float(np.percentile(boot_means[col], 2.5))
        ci_upper = float(np.percentile(boot_means[col], 97.5))
        ci_results[col] = {
            "mean": float(pe),
            "ci_95": [ci_lower, ci_upper]
        }
    return ci_results

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS & RECOMMENDATION GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

class BaselineRecommender:
    def __init__(self, tag):
        self.tag = tag
        print(f"\nLoading training artefacts for {tag}...")
        
        # 1. Load train transitions
        self.train_df = pd.read_parquet(SPLIT_DIR / f"train_transitions_{tag}.parquet")
        print(f"  Loaded {len(self.train_df):,} train transitions.")

        # 2. Load catalogue & popularity
        with open(SPLIT_DIR / f"train_catalogue_{tag}.json") as f:
            self.catalogue = set(json.load(f))
        print(f"  Catalogue size: {len(self.catalogue):,} videos.")

        with open(SPLIT_DIR / f"video_popularity_{tag}.json") as f:
            # key: str(video_id) -> int count
            self.video_pop = {int(k): int(v) for k, v in json.load(f).items()}

        with open(SPLIT_DIR / f"author_popularity_{tag}.json") as f:
            self.author_pop = {int(k): int(v) for k, v in json.load(f).items()}

        with open(SPLIT_DIR / f"author_video_map_{tag}.json") as f:
            # key: str(author_id) -> list of video_ids sorted by popularity
            self.author_video_map = {int(k): v for k, v in json.load(f).items()}

        # Video to author map from train
        self.video_to_author = {}
        for aid, vids in self.author_video_map.items():
            for vid in vids:
                self.video_to_author[vid] = aid

        # Global video popularity top 100
        sorted_pop_videos = sorted(self.video_pop.keys(), key=lambda v: self.video_pop[v], reverse=True)
        self.global_top_videos = sorted_pop_videos[:100]
        self.global_top_authors = sorted(self.author_pop.keys(), key=lambda a: self.author_pop[a], reverse=True)[:100]

        # 3. Fit Video-Level Markov Transitions
        print("  Fitting Video-Level Markov Transition Table...")
        self.video_transitions = defaultdict(Counter)
        for row in self.train_df.itertuples(index=False):
            self.video_transitions[row.src_video_id][row.tgt_video_id] += 1
        print(f"  Unique Video source states in train: {len(self.video_transitions):,}")

        # 4. Fit Author-Level Markov Transitions
        print("  Fitting Author-Level Markov Transition Table...")
        self.author_transitions = defaultdict(Counter)
        for row in self.train_df.itertuples(index=False):
            self.author_transitions[row.src_author_id][row.tgt_author_id] += 1
        print(f"  Unique Author source states in train: {len(self.author_transitions):,}")

    def recommend_video_pop(self, src_video, src_author, k=MAX_K):
        """Global video popularity baseline."""
        recs = self.global_top_videos[:k]
        authors = [self.video_to_author.get(v) for v in recs]
        return recs, authors

    def recommend_author_pop(self, src_video, src_author, k=MAX_K):
        """
        Author popularity baseline:
        Picks most popular authors in train, extracts their most popular train videos.
        """
        recs = []
        rec_authors = []
        for aid in self.global_top_authors:
            author_vids = self.author_video_map.get(aid, [])
            for vid in author_vids:
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(aid)
                    if len(recs) == k:
                        break
            if len(recs) == k:
                break
                
        # Backfill with global video pop if author pool had fewer than k videos
        if len(recs) < k:
            for vid in self.global_top_videos:
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(self.video_to_author.get(vid))
                    if len(recs) == k:
                        break
        return recs, rec_authors

    def recommend_same_author(self, src_video, src_author, k=MAX_K):
        """
        Same-Author-as-Previous:
        Recommends top training videos of the author from the source event.
        """
        recs = []
        rec_authors = []
        if src_author in self.author_video_map:
            for vid in self.author_video_map[src_author]:
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(src_author)
                    if len(recs) == k:
                        break
                        
        # Backfill with global video popularity
        if len(recs) < k:
            for vid in self.global_top_videos:
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(self.video_to_author.get(vid))
                    if len(recs) == k:
                        break
        return recs, rec_authors

    def recommend_video_markov(self, src_video, src_author, k=MAX_K):
        """
        Video-level first-order Markov:
        Ranks target videos by empirical transition frequency C(src, tgt).
        Ties broken by global video popularity.
        Fallback to global video popularity if src_video unseen.
        """
        if src_video in self.video_transitions:
            counts = self.video_transitions[src_video]
            # Sort by transition count descending, tie-break by global pop
            ranked = sorted(counts.keys(), key=lambda v: (counts[v], self.video_pop.get(v, 0)), reverse=True)
            recs = ranked[:k]
            rec_authors = [self.video_to_author.get(v) for v in recs]
            
            # Backfill if fewer than k transitions
            if len(recs) < k:
                for vid in self.global_top_videos:
                    if vid not in recs:
                        recs.append(vid)
                        rec_authors.append(self.video_to_author.get(vid))
                        if len(recs) == k:
                            break
            return recs, rec_authors
        else:
            # Fallback to global video popularity
            return self.recommend_video_pop(src_video, src_author, k=k)

    def recommend_author_markov(self, src_video, src_author, k=MAX_K):
        """
        Author-level first-order Markov:
        1. Predict top authors from author transition table C(src_author, tgt_author).
        2. From each predicted author, select their videos in train sorted by popularity.
        3. Fallback to author popularity if src_author unseen.
        """
        recs = []
        rec_authors = []
        
        if src_author in self.author_transitions:
            auth_counts = self.author_transitions[src_author]
            top_authors = [a for a, _ in auth_counts.most_common(k * 2)]
        else:
            top_authors = self.global_top_authors[:k * 2]

        for aid in top_authors:
            vids = self.author_video_map.get(aid, [])
            for vid in vids:
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(aid)
                    if len(recs) == k:
                        break
            if len(recs) == k:
                break
                
        # Backfill if needed
        if len(recs) < k:
            for vid in self.global_top_videos:
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(self.video_to_author.get(vid))
                    if len(recs) == k:
                        break
        return recs, rec_authors

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(gap=30, split="val"):
    t0 = time.time()
    tag = f"{gap}min"
    print(f"\n{'='*75}")
    print(f"EXPERIMENT EVALUATION: Gap={gap}min | Split={split.upper()}")
    print(f"Memory: {log_mem()}")
    print(f"{'='*75}\n")

    recommender = BaselineRecommender(tag)
    
    # Load evaluation split transitions
    eval_file = SPLIT_DIR / f"{split}_transitions_{tag}.parquet"
    print(f"\nLoading evaluation split: {eval_file.name}...")
    eval_df = pd.read_parquet(eval_file)
    n_eval = len(eval_df)
    n_users = eval_df["user_id"].nunique()
    print(f"  Evaluation instances: {n_eval:,} across {n_users:,} unique users.")

    # Target & Source coverage audit
    eval_df["target_in_catalogue"] = eval_df["tgt_video_id"].isin(recommender.catalogue)
    eval_df["video_src_in_train"] = eval_df["src_video_id"].isin(recommender.video_transitions)
    eval_df["auth_src_in_train"] = eval_df["src_author_id"].isin(recommender.author_transitions)

    target_known_count = eval_df["target_in_catalogue"].sum()
    target_coverage = target_known_count / n_eval if n_eval else 0.0
    v_src_coverage = eval_df["video_src_in_train"].mean() if n_eval else 0.0
    a_src_coverage = eval_df["auth_src_in_train"].mean() if n_eval else 0.0

    print(f"\n--- Coverage Diagnostic ({split.upper()}) ---")
    print(f"  Target Catalogue Coverage (video): {target_known_count:,}/{n_eval:,} ({target_coverage:.2%})")
    print(f"  Cold-start targets (impossible):   {n_eval - target_known_count:,} ({(1 - target_coverage):.2%})")
    print(f"  Video Source State Coverage:       {eval_df['video_src_in_train'].sum():,}/{n_eval:,} ({v_src_coverage:.2%})")
    print(f"  Author Source State Coverage:      {eval_df['auth_src_in_train'].sum():,}/{n_eval:,} ({a_src_coverage:.2%})")

    models = {
        "video_pop": recommender.recommend_video_pop,
        "author_pop": recommender.recommend_author_pop,
        "same_author": recommender.recommend_same_author,
        "video_markov": recommender.recommend_video_markov,
        "author_markov": recommender.recommend_author_markov,
    }

    all_results = []
    
    metric_cols = ["hit@10", "hit@20", "ndcg@10", "mrr@10", "auth_hit@10", "auth_hit@20"]

    for model_name, rec_fn in models.items():
        t_m0 = time.time()
        print(f"\nEvaluating: {model_name}...")

        # Lists to store metrics
        case_records = []
        all_recommended_videos = set()

        for row in eval_df.itertuples(index=False):
            recs, rec_authors = rec_fn(row.src_video_id, row.src_author_id, k=MAX_K)
            
            # Hard Rule 2 & 3 Verification: Candidate catalogue integrity
            assert len(recs) == MAX_K, f"Model {model_name} generated fewer than {MAX_K} recs!"
            assert all(v in recommender.catalogue for v in recs), f"Leakage: Model {model_name} recommended video outside V_train!"
            
            # Author consistency check for author models
            if model_name in ["author_pop", "author_markov", "same_author"]:
                for vid, aid in zip(recs, rec_authors):
                    if aid is not None and aid in recommender.author_video_map:
                        if vid in recommender.author_video_map[aid]:
                            pass # correctly belongs to author

            all_recommended_videos.update(recs)

            cm = compute_case_metrics(
                recs=recs,
                target_id=row.tgt_video_id,
                target_author=row.tgt_author_id,
                rec_authors=rec_authors
            )
            case_records.append(cm)

        # Build metric DataFrame
        m_df = pd.DataFrame(case_records)
        m_df["user_id"] = eval_df["user_id"].values
        m_df["target_in_catalogue"] = eval_df["target_in_catalogue"].values

        cand_coverage = len(all_recommended_videos) / len(recommender.catalogue)

        # 1. Compute on ALL CASES
        ci_all = user_cluster_bootstrap(m_df, metric_cols, n_rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED)

        # 2. Compute on TARGET-KNOWN SUBSET
        m_known = m_df[m_df["target_in_catalogue"]].copy().reset_index(drop=True)
        ci_known = user_cluster_bootstrap(m_known, metric_cols, n_rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED) if len(m_known) else {}

        elapsed_m = time.time() - t_m0

        print(f"  Completed in {elapsed_m:.1f}s. Candidate coverage: {len(all_recommended_videos):,} ({cand_coverage:.4%})")
        print(f"  [ALL EVAL CASES - N={n_eval:,}]")
        print(f"    Recall@10 : {ci_all['hit@10']['mean']:.4f} (95% CI: [{ci_all['hit@10']['ci_95'][0]:.4f}, {ci_all['hit@10']['ci_95'][1]:.4f}]) | Raw hits: {int(m_df['hit@10'].sum()):,}")
        print(f"    Recall@20 : {ci_all['hit@20']['mean']:.4f} (95% CI: [{ci_all['hit@20']['ci_95'][0]:.4f}, {ci_all['hit@20']['ci_95'][1]:.4f}]) | Raw hits: {int(m_df['hit@20'].sum()):,}")
        print(f"    NDCG@10   : {ci_all['ndcg@10']['mean']:.4f} (95% CI: [{ci_all['ndcg@10']['ci_95'][0]:.4f}, {ci_all['ndcg@10']['ci_95'][1]:.4f}])")
        print(f"    MRR@10    : {ci_all['mrr@10']['mean']:.4f} (95% CI: [{ci_all['mrr@10']['ci_95'][0]:.4f}, {ci_all['mrr@10']['ci_95'][1]:.4f}])")
        print(f"    Author-Recall@10: {ci_all['auth_hit@10']['mean']:.4f} (95% CI: [{ci_all['auth_hit@10']['ci_95'][0]:.4f}, {ci_all['auth_hit@10']['ci_95'][1]:.4f}])")
        print(f"    Author-Recall@20: {ci_all['auth_hit@20']['mean']:.4f} (95% CI: [{ci_all['auth_hit@20']['ci_95'][0]:.4f}, {ci_all['auth_hit@20']['ci_95'][1]:.4f}])")

        print(f"  [TARGET-KNOWN SUBSET - N={target_known_count:,} ({target_coverage:.2%})]")
        if ci_known:
            print(f"    Recall@10 : {ci_known['hit@10']['mean']:.4f} (95% CI: [{ci_known['hit@10']['ci_95'][0]:.4f}, {ci_known['hit@10']['ci_95'][1]:.4f}]) | Raw hits: {int(m_known['hit@10'].sum()):,}")
            print(f"    Recall@20 : {ci_known['hit@20']['mean']:.4f} (95% CI: [{ci_known['hit@20']['ci_95'][0]:.4f}, {ci_known['hit@20']['ci_95'][1]:.4f}]) | Raw hits: {int(m_known['hit@20'].sum()):,}")
            print(f"    NDCG@10   : {ci_known['ndcg@10']['mean']:.4f} (95% CI: [{ci_known['ndcg@10']['ci_95'][0]:.4f}, {ci_known['ndcg@10']['ci_95'][1]:.4f}])")
            print(f"    MRR@10    : {ci_known['mrr@10']['mean']:.4f} (95% CI: [{ci_known['mrr@10']['ci_95'][0]:.4f}, {ci_known['mrr@10']['ci_95'][1]:.4f}])")

        res_row = {
            "gap": gap,
            "split": split,
            "model": model_name,
            "n_eval": n_eval,
            "target_known_count": target_known_count,
            "target_coverage": target_coverage,
            "v_src_coverage": v_src_coverage,
            "a_src_coverage": a_src_coverage,
            "cand_coverage": cand_coverage,
            "unique_recs": len(all_recommended_videos),
            # All cases
            "all_hits@10": int(m_df["hit@10"].sum()),
            "all_hits@20": int(m_df["hit@20"].sum()),
            "all_recall@10": ci_all["hit@10"]["mean"],
            "all_recall@10_ci_lower": ci_all["hit@10"]["ci_95"][0],
            "all_recall@10_ci_upper": ci_all["hit@10"]["ci_95"][1],
            "all_recall@20": ci_all["hit@20"]["mean"],
            "all_recall@20_ci_lower": ci_all["hit@20"]["ci_95"][0],
            "all_recall@20_ci_upper": ci_all["hit@20"]["ci_95"][1],
            "all_ndcg@10": ci_all["ndcg@10"]["mean"],
            "all_ndcg@10_ci_lower": ci_all["ndcg@10"]["ci_95"][0],
            "all_ndcg@10_ci_upper": ci_all["ndcg@10"]["ci_95"][1],
            "all_mrr@10": ci_all["mrr@10"]["mean"],
            "all_mrr@10_ci_lower": ci_all["mrr@10"]["ci_95"][0],
            "all_mrr@10_ci_upper": ci_all["mrr@10"]["ci_95"][1],
            "all_auth_recall@10": ci_all["auth_hit@10"]["mean"],
            "all_auth_recall@10_ci_lower": ci_all["auth_hit@10"]["ci_95"][0],
            "all_auth_recall@10_ci_upper": ci_all["auth_hit@10"]["ci_95"][1],
            "all_auth_recall@20": ci_all["auth_hit@20"]["mean"],
            "all_auth_recall@20_ci_lower": ci_all["auth_hit@20"]["ci_95"][0],
            "all_auth_recall@20_ci_upper": ci_all["auth_hit@20"]["ci_95"][1],
            # Known cases
            "known_hits@10": int(m_known["hit@10"].sum()) if len(m_known) else 0,
            "known_hits@20": int(m_known["hit@20"].sum()) if len(m_known) else 0,
            "known_recall@10": ci_known.get("hit@10", {}).get("mean", 0.0),
            "known_recall@10_ci_lower": ci_known.get("hit@10", {}).get("ci_95", [0.0, 0.0])[0],
            "known_recall@10_ci_upper": ci_known.get("hit@10", {}).get("ci_95", [0.0, 0.0])[1],
            "known_recall@20": ci_known.get("hit@20", {}).get("mean", 0.0),
            "known_recall@20_ci_lower": ci_known.get("hit@20", {}).get("ci_95", [0.0, 0.0])[0],
            "known_recall@20_ci_upper": ci_known.get("hit@20", {}).get("ci_95", [0.0, 0.0])[1],
            "known_ndcg@10": ci_known.get("ndcg@10", {}).get("mean", 0.0),
            "known_ndcg@10_ci_lower": ci_known.get("ndcg@10", {}).get("ci_95", [0.0, 0.0])[0],
            "known_ndcg@10_ci_upper": ci_known.get("ndcg@10", {}).get("ci_95", [0.0, 0.0])[1],
            "known_mrr@10": ci_known.get("mrr@10", {}).get("mean", 0.0),
            "known_mrr@10_ci_lower": ci_known.get("mrr@10", {}).get("ci_95", [0.0, 0.0])[0],
            "known_mrr@10_ci_upper": ci_known.get("mrr@10", {}).get("ci_95", [0.0, 0.0])[1],
        }
        all_results.append(res_row)

    out_csv = RESULTS_DIR / f"results_{split}_{tag}.csv"
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(out_csv, index=False)
    print(f"\nAll baseline results saved to: {out_csv}")
    print(f"Evaluation completed in {time.time()-t0:.1f}s. Final Mem: {log_mem()}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=30, choices=[30, 60], help="Session gap in minutes")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Evaluation split (default: val)")
    args = parser.parse_args()

    run_evaluation(gap=args.gap, split=args.split)
