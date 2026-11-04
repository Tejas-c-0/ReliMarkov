"""
estimate_eb.py  —  Empirical Bayes Prior Estimation on Training Transitions
===========================================================================
Estimates exact global optimum alpha_0^EB for:
1. Video-level Markov transitions
2. Author-level Markov transitions
across 30-min and 60-min training splits.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln, psi
from scipy.optimize import minimize_scalar

DATA_DIR = Path(r"D:\APPS Research\KuaiRand-1K\data\splits")
OUTPUT_DIR = Path(r"D:\APPS Research\KuaiRand-1K\outputs\eb_params")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fit_eb_concentration(train_df, src_col, tgt_col, base_measure_dict, label=""):
    print(f"\n--- Estimating EB Prior Concentration: {label} ---")
    
    grouped = train_df.groupby([src_col, tgt_col]).size().reset_index(name="count")
    src_totals = grouped.groupby(src_col)["count"].sum()
    
    N_s_array = src_totals.values.astype(np.float64)
    counts = grouped["count"].values.astype(np.float64)
    m_t = np.array([base_measure_dict.get(t, 1e-9) for t in grouped[tgt_col]], dtype=np.float64)
    m_t = m_t / sum(base_measure_dict.values())

    def neg_log_evidence(alpha0):
        if alpha0 <= 1e-6:
            return 1e15
        term1 = np.sum(gammaln(alpha0) - gammaln(N_s_array + alpha0))
        term2 = np.sum(gammaln(counts + alpha0 * m_t) - gammaln(alpha0 * m_t))
        return -(term1 + term2)

    def d_neg_log_evidence(alpha0):
        term1 = np.sum(psi(alpha0) - psi(N_s_array + alpha0))
        term2 = np.sum(m_t * (psi(counts + alpha0 * m_t) - psi(alpha0 * m_t)))
        return -(term1 + term2)

    # 1D Bounded Brent Optimization with wide bounds [0.1, 50000.0]
    res = minimize_scalar(neg_log_evidence, bounds=(0.1, 50000.0), method="bounded", options={"xatol": 1e-3})
    alpha0_opt = float(res.x)
    min_nll = float(res.fun)
    grad_at_opt = float(d_neg_log_evidence(alpha0_opt))

    print(f"  Optimized alpha_0^EB: {alpha0_opt:.4f}")
    print(f"  Negative Log-Evidence: {min_nll:,.4f}")
    print(f"  Gradient at Optimum  : {grad_at_opt:+.2e}")

    # Boundary inspection
    nll_low = float(neg_log_evidence(0.1))
    nll_high = float(neg_log_evidence(50000.0))
    print(f"  Boundary NLL at 0.1   : {nll_low:,.4f} (higher than opt: {nll_low > min_nll})")
    print(f"  Boundary NLL at 50000 : {nll_high:,.4f} (higher than opt: {nll_high > min_nll})")

    return {
        "alpha0": alpha0_opt,
        "neg_log_evidence": min_nll,
        "gradient": grad_at_opt,
        "n_sources": len(N_s_array),
        "total_transitions": int(np.sum(counts)),
    }

for gap in [30, 60]:
    tag = f"{gap}min"
    print(f"\n{'='*70}\nPROCESSING {tag.upper()}\n{'='*70}")
    train_df = pd.read_parquet(DATA_DIR / f"train_transitions_{tag}.parquet")
    
    with open(DATA_DIR / f"video_popularity_{tag}.json") as f:
        vid_pop = {int(k): int(v) for k, v in json.load(f).items()}
    with open(DATA_DIR / f"author_popularity_{tag}.json") as f:
        auth_pop = {int(k): int(v) for k, v in json.load(f).items()}

    tot_vid = sum(vid_pop.values())
    m_vid = {k: v / tot_vid for k, v in vid_pop.items()}

    tot_auth = sum(auth_pop.values())
    m_auth = {k: v / tot_auth for k, v in auth_pop.items()}

    eb_vid = fit_eb_concentration(
        train_df, src_col="src_video_id", tgt_col="tgt_video_id",
        base_measure_dict=m_vid, label=f"Video-Level ({tag})"
    )

    eb_auth = fit_eb_concentration(
        train_df, src_col="src_author_id", tgt_col="tgt_author_id",
        base_measure_dict=m_auth, label=f"Author-Level ({tag})"
    )

    out_file = OUTPUT_DIR / f"eb_hyperparams_{tag}.json"
    with open(out_file, "w") as f:
        json.dump({"video_eb": eb_vid, "author_eb": eb_auth}, f, indent=2)
    print(f"\nSaved EB parameters to: {out_file}")
