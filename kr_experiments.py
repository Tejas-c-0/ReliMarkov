"""
kr_experiments.py  —  Validation Experiments for M1 to M5 Variants
===================================================================

Models Evaluated on Validation Split:
  - Baselines:
      * video_pop
      * author_pop
      * same_author_backfill (reports direct and backfill hits separately)
  - M1: markov_mle (Standard MLE Markov)
  - M2: markov_dirichlet (Fixed Dirichlet prior: grid search over alpha)
  - M3: markov_eb (Empirical Bayes prior concentration alpha_0^EB)
  - M4: markov_rw (Reliability-weighted Markov: W1, W2, W3, W4 ablations)
  - M5-A: ra_ebmm_raw (Raw W4 pseudo-counts + EB alpha_0^EB)
  - M5-B: ra_ebmm_norm (Scale-normalized W4 to mean 1.0 + EB alpha_0^EB)
  - M5-C: ra_ebmm_calib (Raw W4 + multiplier lambda * alpha_0^EB)

Both 30-min (primary) and 60-min (sensitivity) are evaluated on the VALIDATION split.
The test set is frozen and untouched.
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

DATA_DIR = Path(r"D:\APPS Research\KuaiRand-1K\data\splits")
EB_DIR = Path(r"D:\APPS Research\KuaiRand-1K\outputs\eb_params")
RESULTS_DIR = Path(r"D:\APPS Research\KuaiRand-1K\outputs\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42
MAX_K = 20

def log_mem():
    mb = psutil.Process().memory_info().rss / 1_048_576
    return f"{mb:.0f} MB"

def compute_case_metrics(recs, target_id, target_author, rec_authors=None):
    metrics = {
        "hit@10": 0.0,
        "hit@20": 0.0,
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "auth_hit@10": 0.0,
        "auth_hit@20": 0.0,
    }
    if target_id in recs:
        rank = recs.index(target_id) + 1
        if rank <= 10:
            metrics["hit@10"] = 1.0
            metrics["ndcg@10"] = 1.0 / math.log2(rank + 1)
            metrics["mrr@10"] = 1.0 / rank
        if rank <= 20:
            metrics["hit@20"] = 1.0

    if rec_authors is not None and target_author is not None:
        if target_author in rec_authors[:10]:
            metrics["auth_hit@10"] = 1.0
        if target_author in rec_authors[:20]:
            metrics["auth_hit@20"] = 1.0

    return metrics

def user_cluster_bootstrap(df_eval, metric_cols, n_rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    rng = np.random.RandomState(seed)
    users = df_eval["user_id"].unique()
    n_users = len(users)
    user_to_idx = df_eval.groupby("user_id").indices
    arrays = {col: df_eval[col].to_numpy(dtype=np.float64) for col in metric_cols}
    boot_means = {col: np.empty(n_rounds, dtype=np.float64) for col in metric_cols}
    
    for b in range(n_rounds):
        sampled_users = rng.choice(users, size=n_users, replace=True)
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

class ExperimentEngine:
    def __init__(self, tag="30min"):
        self.tag = tag
        print(f"\n[Initializing ExperimentEngine for {tag}]")
        t0 = time.time()
        
        # 1. Load train transitions
        self.train_df = pd.read_parquet(DATA_DIR / f"train_transitions_{tag}.parquet")
        
        # 2. Load catalogue & popularity structures
        with open(DATA_DIR / f"train_catalogue_{tag}.json") as f:
            self.catalogue = set(json.load(f))
        with open(DATA_DIR / f"video_popularity_{tag}.json") as f:
            self.video_pop = {int(k): int(v) for k, v in json.load(f).items()}
        with open(DATA_DIR / f"author_popularity_{tag}.json") as f:
            self.author_pop = {int(k): int(v) for k, v in json.load(f).items()}
        with open(DATA_DIR / f"author_video_map_{tag}.json") as f:
            self.author_video_map = {int(k): v for k, v in json.load(f).items()}

        self.video_to_author = {}
        for aid, vids in self.author_video_map.items():
            for vid in vids:
                self.video_to_author[vid] = aid

        # Sorted global popular items (strict positive support base measure)
        self.global_top_videos = sorted(self.video_pop.keys(), key=lambda v: self.video_pop[v], reverse=True)[:100]
        self.global_top_authors = sorted(self.author_pop.keys(), key=lambda a: self.author_pop[a], reverse=True)[:100]

        tot_v = sum(self.video_pop.values())
        self.m_vid = {k: v / tot_v for k, v in self.video_pop.items()}
        tot_a = sum(self.author_pop.values())
        self.m_auth = {k: v / tot_a for k, v in self.author_pop.items()}

        # 3. Load EB parameters
        with open(EB_DIR / f"eb_hyperparams_{tag}.json") as f:
            eb_data = json.load(f)
            self.alpha_eb_vid = eb_data["video_eb"]["alpha0"]
            self.alpha_eb_auth = eb_data["author_eb"]["alpha0"]

        print(f"  Loaded EB parameters: Video alpha_0={self.alpha_eb_vid:.4f}, Author alpha_0={self.alpha_eb_auth:.4f}")

        # 4. Load raw interaction signals for reliability weights
        print("  Loading interaction event weights for transitions...")
        cols = ["user_id", "video_id", "time_ms", "play_time_ms", "duration_ms", "long_view", "is_like", "is_forward", "is_hate", "tab", "is_click"]
        f1 = pd.read_csv(Path(r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"), usecols=cols)
        f2 = pd.read_csv(Path(r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_22_to_5_08_1k.csv"), usecols=cols)
        logs = pd.concat([f1, f2], ignore_index=True)
        del f1, f2
        gc.collect()

        logs = logs[(logs["tab"] == 1) & (logs["is_click"] == 1)].copy()
        
        # Duration-zero ablation handling: impute with median watch ratio 0.9016
        dur_pos = logs["duration_ms"] > 0
        wr = np.where(dur_pos, logs["play_time_ms"] / logs["duration_ms"].astype(float), 0.9016)
        logs["watch_ratio"] = np.clip(wr, 0.0, 1.0)
        
        # Event weight formula (excluding is_click which is constant 1 in Dataset A)
        w = logs["watch_ratio"] + 0.3 * logs["is_like"] + 0.2 * logs["is_forward"] - 0.5 * logs["is_hate"]
        logs["event_weight"] = np.clip(w, 0.1, 1.5)

        self.event_weight_dict = logs.set_index(["user_id", "video_id", "time_ms"])["event_weight"].to_dict()
        del logs
        gc.collect()

        # 5. Pre-compute Transition Tables
        print("  Building transition tables...")
        # Unweighted integer counts
        self.C_vid = defaultdict(Counter)
        self.C_auth = defaultdict(Counter)
        
        # Reliability weighted counts W4 (geometric mean)
        self.W4_vid = defaultdict(lambda: defaultdict(float))
        self.W4_auth = defaultdict(lambda: defaultdict(float))
        
        # W2 (source only) and W3 (target only)
        self.W2_vid = defaultdict(lambda: defaultdict(float))
        self.W3_vid = defaultdict(lambda: defaultdict(float))

        w4_values = []
        for r in self.train_df.itertuples(index=False):
            w_s = self.event_weight_dict.get((r.user_id, r.src_video_id, r.src_time_ms), 0.68)
            w_t = self.event_weight_dict.get((r.user_id, r.tgt_video_id, r.tgt_time_ms), 0.68)
            w4 = math.sqrt(w_s * w_t)
            w4_values.append(w4)

            # Integer
            self.C_vid[r.src_video_id][r.tgt_video_id] += 1
            self.C_auth[r.src_author_id][r.tgt_author_id] += 1

            # Weighted
            self.W4_vid[r.src_video_id][r.tgt_video_id] += w4
            self.W4_auth[r.src_author_id][r.tgt_author_id] += w4
            self.W2_vid[r.src_video_id][r.tgt_video_id] += w_s
            self.W3_vid[r.src_video_id][r.tgt_video_id] += w_t

        self.mean_w4_train = float(np.mean(w4_values))
        print(f"  Mean W4 in train: {self.mean_w4_train:.4f} (used for M5-B normalization). Mem: {log_mem()}")

    def get_query_weight(self, user_id, src_video_id, src_time_ms):
        return self.event_weight_dict.get((user_id, src_video_id, src_time_ms), 0.68)

    # ─────────────────────────────────────────────────────────────────────────
    # Model Implementations
    # ─────────────────────────────────────────────────────────────────────────

    def rec_m1_mle(self, src_video, src_author, user_id, src_time):
        """M1: Standard Markov MLE with global popularity fallback."""
        if src_video in self.C_vid:
            counts = self.C_vid[src_video]
            ranked = sorted(counts.keys(), key=lambda v: (counts[v], self.video_pop.get(v, 0)), reverse=True)[:MAX_K]
            recs = ranked.copy()
            if len(recs) < MAX_K:
                for v in self.global_top_videos:
                    if v not in recs:
                        recs.append(v)
                        if len(recs) == MAX_K:
                            break
            return recs, [self.video_to_author.get(v) for v in recs]
        else:
            return self.global_top_videos[:MAX_K], [self.video_to_author.get(v) for v in self.global_top_videos[:MAX_K]]

    def rec_m2_dirichlet(self, src_video, src_author, user_id, src_time, alpha=1.0):
        """M2: Fixed Dirichlet-smoothed Markov."""
        counts = self.C_vid.get(src_video, Counter())
        n_s = sum(counts.values())
        # Score candidates from observed transitions + top global popularity
        candidates = set(counts.keys()) | set(self.global_top_videos[:50])
        # Score(v) = (C(s, v) + alpha * m_v) / (N_s + alpha)
        # Note: denominator (N_s + alpha) is constant across all candidates for a given query!
        scored = [(v, counts[v] + alpha * self.m_vid.get(v, 1e-9)) for v in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recs = [v for v, s in scored[:MAX_K]]
        return recs, [self.video_to_author.get(v) for v in recs]

    def rec_m3_eb(self, src_video, src_author, user_id, src_time):
        """M3: Empirical Bayes smoothed Markov (alpha = alpha_0^EB)."""
        return self.rec_m2_dirichlet(src_video, src_author, user_id, src_time, alpha=self.alpha_eb_vid)

    def rec_m4_rw(self, src_video, src_author, user_id, src_time, weighting_scheme="W4", alpha=1.0):
        """M4: Reliability-Weighted Markov."""
        if weighting_scheme == "W4":
            table = self.W4_vid.get(src_video, {})
        elif weighting_scheme == "W2":
            table = self.W2_vid.get(src_video, {})
        elif weighting_scheme == "W3":
            table = self.W3_vid.get(src_video, {})
        else: # W1 unweighted
            table = self.C_vid.get(src_video, {})

        w_q = self.get_query_weight(user_id, src_video, src_time)
        candidates = set(table.keys()) | set(self.global_top_videos[:50])
        scored = [(v, w_q * table.get(v, 0.0) + alpha * self.m_vid.get(v, 1e-9)) for v in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recs = [v for v, s in scored[:MAX_K]]
        return recs, [self.video_to_author.get(v) for v in recs]

    def rec_m5a_raw(self, src_video, src_author, user_id, src_time):
        """M5-A: Raw W4 pseudo-counts + raw EB alpha_0^EB."""
        table = self.W4_vid.get(src_video, {})
        w_q = self.get_query_weight(user_id, src_video, src_time)
        candidates = set(table.keys()) | set(self.global_top_videos[:50])
        scored = [(v, w_q * table.get(v, 0.0) + self.alpha_eb_vid * self.m_vid.get(v, 1e-9)) for v in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recs = [v for v, s in scored[:MAX_K]]
        return recs, [self.video_to_author.get(v) for v in recs]

    def rec_m5b_norm(self, src_video, src_author, user_id, src_time):
        """M5-B: Scale-Normalized W4 (mean=1.0) + raw EB alpha_0^EB."""
        table = self.W4_vid.get(src_video, {})
        mu = self.mean_w4_train
        w_q_norm = self.get_query_weight(user_id, src_video, src_time) / mu
        candidates = set(table.keys()) | set(self.global_top_videos[:50])
        scored = [(v, w_q_norm * (table.get(v, 0.0) / mu) + self.alpha_eb_vid * self.m_vid.get(v, 1e-9)) for v in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recs = [v for v, s in scored[:MAX_K]]
        return recs, [self.video_to_author.get(v) for v in recs]

    def rec_m5c_calib(self, src_video, src_author, user_id, src_time, lamb=0.5543):
        """M5-C: Raw W4 + calibrated multiplier lambda * alpha_0^EB."""
        table = self.W4_vid.get(src_video, {})
        w_q = self.get_query_weight(user_id, src_video, src_time)
        effective_alpha = lamb * self.alpha_eb_vid
        candidates = set(table.keys()) | set(self.global_top_videos[:50])
        scored = [(v, w_q * table.get(v, 0.0) + effective_alpha * self.m_vid.get(v, 1e-9)) for v in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recs = [v for v, s in scored[:MAX_K]]
        return recs, [self.video_to_author.get(v) for v in recs]

    # Author-level models
    def rec_author_eb(self, src_video, src_author, user_id, src_time, alpha=None):
        if alpha is None:
            alpha = self.alpha_eb_auth
        counts = self.C_auth.get(src_author, Counter())
        candidates = set(counts.keys()) | set(self.global_top_authors[:50])
        scored = [(a, counts[a] + alpha * self.m_auth.get(a, 1e-9)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Retrieve videos of top authors
        recs = []
        rec_authors = []
        for aid, s in scored:
            for vid in self.author_video_map.get(aid, []):
                if vid not in recs:
                    recs.append(vid)
                    rec_authors.append(aid)
                    if len(recs) == MAX_K:
                        break
            if len(recs) == MAX_K:
                break
        if len(recs) < MAX_K:
            for v in self.global_top_videos:
                if v not in recs:
                    recs.append(v)
                    rec_authors.append(self.video_to_author.get(v))
                    if len(recs) == MAX_K:
                        break
        return recs, rec_authors

def run_all_validation(gap=30):
    tag = f"{gap}min"
    print(f"\n{'='*75}\nRUNNING COMPREHENSIVE VALIDATION EXPERIMENTS: {tag.upper()}\n{'='*75}")
    engine = ExperimentEngine(tag)
    
    val_file = DATA_DIR / f"val_transitions_{tag}.parquet"
    val_df = pd.read_parquet(val_file)
    n_val = len(val_df)
    val_df["target_in_catalogue"] = val_df["tgt_video_id"].isin(engine.catalogue)

    models_to_test = [
        ("M1_markov_mle", lambda sv, sa, u, st: engine.rec_m1_mle(sv, sa, u, st)),
        ("M2_dirichlet_alpha_1.0", lambda sv, sa, u, st: engine.rec_m2_dirichlet(sv, sa, u, st, alpha=1.0)),
        ("M2_dirichlet_alpha_10.0", lambda sv, sa, u, st: engine.rec_m2_dirichlet(sv, sa, u, st, alpha=10.0)),
        ("M2_dirichlet_alpha_50.0", lambda sv, sa, u, st: engine.rec_m2_dirichlet(sv, sa, u, st, alpha=50.0)),
        ("M3_markov_eb", lambda sv, sa, u, st: engine.rec_m3_eb(sv, sa, u, st)),
        ("M4_rw_W1_unweighted", lambda sv, sa, u, st: engine.rec_m4_rw(sv, sa, u, st, weighting_scheme="W1", alpha=1.0)),
        ("M4_rw_W2_src_only", lambda sv, sa, u, st: engine.rec_m4_rw(sv, sa, u, st, weighting_scheme="W2", alpha=1.0)),
        ("M4_rw_W3_tgt_only", lambda sv, sa, u, st: engine.rec_m4_rw(sv, sa, u, st, weighting_scheme="W3", alpha=1.0)),
        ("M4_rw_W4_geom", lambda sv, sa, u, st: engine.rec_m4_rw(sv, sa, u, st, weighting_scheme="W4", alpha=1.0)),
        ("M5A_ra_ebmm_raw", lambda sv, sa, u, st: engine.rec_m5a_raw(sv, sa, u, st)),
        ("M5B_ra_ebmm_norm", lambda sv, sa, u, st: engine.rec_m5b_norm(sv, sa, u, st)),
        ("M5C_ra_ebmm_calib_0.55", lambda sv, sa, u, st: engine.rec_m5c_calib(sv, sa, u, st, lamb=engine.mean_w4_train)),
        ("Author_EB", lambda sv, sa, u, st: engine.rec_author_eb(sv, sa, u, st)),
    ]

    metric_cols = ["hit@10", "hit@20", "ndcg@10", "mrr@10", "auth_hit@10", "auth_hit@20"]
    all_summary = []

    for name, rec_fn in models_to_test:
        t0 = time.time()
        records = []
        all_recs = set()
        
        for r in val_df.itertuples(index=False):
            recs, rec_authors = rec_fn(r.src_video_id, r.src_author_id, r.user_id, r.src_time_ms)
            assert len(recs) == MAX_K
            assert all(v in engine.catalogue for v in recs)
            all_recs.update(recs)
            cm = compute_case_metrics(recs, r.tgt_video_id, r.tgt_author_id, rec_authors)
            records.append(cm)

        m_df = pd.DataFrame(records)
        m_df["user_id"] = val_df["user_id"].values
        m_df["target_in_catalogue"] = val_df["target_in_catalogue"].values
        m_known = m_df[m_df["target_in_catalogue"]].reset_index(drop=True)

        ci_all = user_cluster_bootstrap(m_df, metric_cols)
        ci_known = user_cluster_bootstrap(m_known, metric_cols) if len(m_known) else {}

        elapsed = time.time() - t0
        print(f"\n[{name}] completed in {elapsed:.1f}s | Unique Recs: {len(all_recs):,}")
        print(f"  All Val:   Recall@10={ci_all['hit@10']['mean']:.6f} (hits={int(m_df['hit@10'].sum())}) | Recall@20={ci_all['hit@20']['mean']:.6f} (hits={int(m_df['hit@20'].sum())})")
        print(f"  Known Sub: Recall@10={ci_known['hit@10']['mean']:.6f} | Recall@20={ci_known['hit@20']['mean']:.6f}")
        print(f"  Author-Recall@20: {ci_all['auth_hit@20']['mean']:.6f} (hits={int(m_df['auth_hit@20'].sum())})")

        row = {
            "gap": gap,
            "model": name,
            "elapsed_sec": round(elapsed, 1),
            "unique_recs": len(all_recs),
            "cand_coverage": len(all_recs) / len(engine.catalogue),
            # All
            "all_hits@10": int(m_df["hit@10"].sum()),
            "all_hits@20": int(m_df["hit@20"].sum()),
            "all_recall@10": ci_all["hit@10"]["mean"],
            "all_recall@10_ci_lower": ci_all["hit@10"]["ci_95"][0],
            "all_recall@10_ci_upper": ci_all["hit@10"]["ci_95"][1],
            "all_recall@20": ci_all["hit@20"]["mean"],
            "all_recall@20_ci_lower": ci_all["hit@20"]["ci_95"][0],
            "all_recall@20_ci_upper": ci_all["hit@20"]["ci_95"][1],
            "all_ndcg@10": ci_all["ndcg@10"]["mean"],
            "all_mrr@10": ci_all["mrr@10"]["mean"],
            "all_auth_recall@10": ci_all["auth_hit@10"]["mean"],
            "all_auth_recall@20": ci_all["auth_hit@20"]["mean"],
            # Known
            "known_hits@10": int(m_known["hit@10"].sum()),
            "known_hits@20": int(m_known["hit@20"].sum()),
            "known_recall@10": ci_known["hit@10"]["mean"],
            "known_recall@20": ci_known["hit@20"]["mean"],
            "known_ndcg@10": ci_known["ndcg@10"]["mean"],
            "known_mrr@10": ci_known["mrr@10"]["mean"],
        }
        all_summary.append(row)

    out_csv = RESULTS_DIR / f"val_experiments_{tag}.csv"
    pd.DataFrame(all_summary).to_csv(out_csv, index=False)
    print(f"\nSaved all experiment results to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=30, choices=[30, 60])
    args = parser.parse_args()
    run_all_validation(gap=args.gap)
