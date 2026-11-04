"""
00_profile.py  —  KuaiRand-1K Stage-0 Data Profiling
=====================================================

PURPOSE
  Pure diagnostic.  Establishes data facts needed to design the experiment.
  No 5-core filtering.  No chronological split.  No model training.
  No reliability weighting.  No artefacts written to disk.

DATASETS COMPARED
  Dataset A : tab == 1  AND  is_click == 1
  Dataset B : tab == 1  (all valid-play interactions on the main feed)

TAB / IS_CLICK SEMANTICS  (verified from official KuaiRand README)
  tab == 1  : Main recommendation page, single-column UI.
  is_click  : On the single-column UI this field records *valid_play*, NOT a
              screen tap.  Valid play equals 1 when:
                play_duration >= video_duration  if video_duration <= 7 s
                play_duration > 7 s              if video_duration >  7 s
              This is a stronger engagement filter than a tap.

TIMEZONE ASSUMPTION
  The raw 'date' column (format YYYYMMDD) is in Beijing time (UTC+8 / CST)
  because Kuaishou is a Chinese app.  We use this column directly for daily
  aggregations rather than converting time_ms, to avoid any ambiguity.
  time_ms is Unix epoch milliseconds (UTC).  Any time_ms-derived dates would
  need a +8 h offset to match the 'date' column.

REQUIREMENTS
  pip install pandas numpy psutil
  (psutil is optional; the script degrades gracefully without it.)

USAGE
  cd "D:\\APPS Research\\KuaiRand-1K"
  python 00_profile.py 2>&1 | Tee-Object -FilePath profile_run.txt
"""

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR       = Path(r"D:\APPS Research\KuaiRand-1K\data")
LOG_FILE_1     = DATA_DIR / "log_standard_4_08_to_4_21_1k.csv"
LOG_FILE_2     = DATA_DIR / "log_standard_4_22_to_5_08_1k.csv"
VIDEO_META     = DATA_DIR / "video_features_basic_1k.csv"

# Session gaps (milliseconds) to test in the conservative transition counts.
SESSION_GAPS = {"30min": 30 * 60 * 1_000,
                "60min": 60 * 60 * 1_000}

# Columns we actually need from the log files (read nothing else).
LOG_COLS   = ["user_id", "video_id", "date", "time_ms",
              "is_click", "is_like", "is_forward", "is_hate",
              "long_view", "play_time_ms", "duration_ms", "tab"]

# Explicit dtypes — keeps memory tight and prevents silent widening.
LOG_DTYPES = {
    "user_id":      "int32",   # 1 000 users in the 1K version
    "video_id":     "int64",   # up to ~4.3 M unique videos
    "date":         "int32",   # YYYYMMDD  e.g. 20220409
    "time_ms":      "int64",   # Unix ms timestamp
    "is_click":     "int8",    # binary 0/1
    "is_like":      "int8",
    "is_forward":   "int8",
    "is_hate":      "int8",
    "long_view":    "int8",
    "play_time_ms": "int64",
    "duration_ms":  "int64",
    "tab":          "int8",    # range 0–14
}

# Metadata: only these two columns are needed.
META_COLS = ["video_id", "author_id"]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

try:
    import psutil
    def _rss_mb() -> float:
        return psutil.Process().memory_info().rss / 1_048_576
except ImportError:
    def _rss_mb() -> float:
        return float("nan")

_t_start = time.time()

def _elapsed() -> str:
    return f"{time.time() - _t_start:6.1f}s"

def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
    print(f"  elapsed={_elapsed()}  mem={_rss_mb():.0f} MB")
    sys.stdout.flush()

def _pct(n: int, d: int) -> str:
    return f"{n/d:.4f}" if d else "N/A"

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FILE EXISTENCE CHECK
# ─────────────────────────────────────────────────────────────────────────────

section("FILE EXISTENCE CHECK")

for p in [LOG_FILE_1, LOG_FILE_2, VIDEO_META]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    mb = p.stat().st_size / 1_048_576
    print(f"  {p.name:50s}  {mb:7.0f} MB  [OK]")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  VIDEO METADATA — LOAD AND VALIDATE AUTHOR MAPPING
# ─────────────────────────────────────────────────────────────────────────────

section("VIDEO METADATA — LOAD (video_id + author_id only)")

t0 = time.time()
vf_raw = pd.read_csv(
    VIDEO_META,
    usecols=META_COLS,
    # author_id may have NaN in the CSV -> let pandas infer float64, then convert
    dtype={"video_id": "int64"},
    engine="c",
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Rows            : {len(vf_raw):,}")
print(f"  Unique video_ids: {vf_raw['video_id'].nunique():,}")

# Coerce author_id to numeric; anything that fails -> NaN
vf_raw["author_id"] = pd.to_numeric(vf_raw["author_id"], errors="coerce")

n_has_author    = vf_raw["author_id"].notna().sum()
n_miss_author   = vf_raw["author_id"].isna().sum()
print(f"  Rows with author_id   : {n_has_author:,}")
print(f"  Rows missing author_id: {n_miss_author:,}")

# ── HARD RULE 6: Raise on conflicting video -> author mappings ────────────────
section("AUTHOR MAPPING VALIDATION (HARD RULE 6)")

dup_vids = vf_raw.duplicated(subset=["video_id"], keep=False)
n_dup_vid_rows = dup_vids.sum()
print(f"  Rows sharing a video_id with another row: {n_dup_vid_rows:,}")

if n_dup_vid_rows > 0:
    # Check whether any video maps to more than one distinct non-null author
    conflict_check = (
        vf_raw.dropna(subset=["author_id"])
        .groupby("video_id")["author_id"]
        .nunique()
    )
    conflicting = conflict_check[conflict_check > 1]
    n_conflicts = len(conflicting)
    print(f"  Videos with conflicting author_id values: {n_conflicts:,}")

    if n_conflicts > 0:
        print("\n  First 20 conflicting video -> author mappings:")
        for vid in conflicting.index[:20]:
            rows = vf_raw[vf_raw["video_id"] == vid]
            authors = rows["author_id"].dropna().unique().tolist()
            print(f"    video_id={vid}: author_ids={authors}")
        raise ValueError(
            f"\n*** HARD RULE 6 VIOLATION ***\n"
            f"{n_conflicts} video(s) map to more than one author_id in "
            f"video_features_basic_1k.csv.\n"
            f"Inspect the metadata file and resolve conflicts before continuing."
        )
    else:
        print("  Duplicate rows exist but all share the same author_id: NO CONFLICT.")
else:
    print("  No duplicate video_id rows found in metadata: mapping is one-to-one.")

# ── Build the clean unambiguous mapping ──────────────────────────────────────
author_map = (
    vf_raw.dropna(subset=["author_id"])
    .drop_duplicates(subset=["video_id"], keep="first")
    [["video_id", "author_id"]]
    .copy()
)
# Safe to cast now that conflicts are ruled out
author_map["author_id"] = author_map["author_id"].astype("int64")

print(f"\n  Clean video -> author entries : {len(author_map):,}")
print(f"  Unique authors in mapping    : {author_map['author_id'].nunique():,}")
print(f"  Videos WITHOUT author mapping: "
      f"{vf_raw['video_id'].nunique() - len(author_map):,}")

del vf_raw
gc.collect()
print(f"  mem after metadata free: {_rss_mb():.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOAD LOG FILES
# ─────────────────────────────────────────────────────────────────────────────

def _load_log(path: Path, tag: str) -> pd.DataFrame:
    section(f"LOG FILE — {tag}  ({path.name})")
    t0 = time.time()
    df = pd.read_csv(
        path,
        usecols=LOG_COLS,
        dtype=LOG_DTYPES,
        engine="c",
    )
    df["_src"] = tag          # track which file each row came from
    elapsed = time.time() - t0
    print(f"  Rows loaded     : {len(df):,}")
    print(f"  Load time       : {elapsed:.1f}s")
    print(f"  Memory now      : {_rss_mb():.0f} MB")
    print(f"  date range      : {df['date'].min()} – {df['date'].max()}")
    t_lo = pd.Timestamp(df["time_ms"].min(), unit="ms", tz="UTC")
    t_hi = pd.Timestamp(df["time_ms"].max(), unit="ms", tz="UTC")
    print(f"  time_ms -> UTC   : {t_lo}  –  {t_hi}")
    print(f"  Users           : {df['user_id'].nunique():,}")
    print(f"  Videos          : {df['video_id'].nunique():,}")
    print(f"  is_rand==1 rows : {(df.get('is_rand',pd.Series(dtype=int))==1).sum():,}"
          f"  (randomly-exposed items embedded in standard feed)")
    return df

log1 = _load_log(LOG_FILE_1, "file1")
log2 = _load_log(LOG_FILE_2, "file2")

# Verify the two files do not have overlapping dates
date_overlap = set(log1["date"].unique()) & set(log2["date"].unique())
section("DATE-RANGE OVERLAP CHECK")
if date_overlap:
    print(f"  WARNING: {len(date_overlap)} date(s) appear in BOTH log files:")
    print(f"  {sorted(date_overlap)}")
    print(f"  This may indicate legitimate re-collection or a data issue.")
    print(f"  Cross-file duplicate check (next section) will quantify exact overlaps.")
else:
    print(f"  No overlapping dates between the two files.  [EXPECTED]")
    print(f"  file1 ends   : {log1['date'].max()}")
    print(f"  file2 begins : {log2['date'].min()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-FILE DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

section("CROSS-FILE DUPLICATE DETECTION")
print("  A potential duplicate = same (user_id, video_id, time_ms) in both files.")
print("  We report counts only.  Nothing is deleted.")

keys = ["user_id", "video_id", "time_ms"]

k1 = log1[keys].drop_duplicates()
k2 = log2[keys].drop_duplicates()

# Inner join on three key columns
overlap_df = pd.merge(k1, k2, on=keys, how="inner")
n_overlap = len(overlap_df)

print(f"  Unique triples in file1         : {len(k1):,}")
print(f"  Unique triples in file2         : {len(k2):,}")
print(f"  Overlapping (potential dups)    : {n_overlap:,}")
if n_overlap > 0:
    print(f"\n  WARNING: {n_overlap:,} triples appear in both log files.")
    print(f"  Investigate before building the final pipeline.")
    print(f"  First 5 overlapping triples:")
    print(overlap_df.head(5).to_string(index=False))
else:
    print(f"  No cross-file duplicates on (user_id, video_id, time_ms).  [OK]")

del k1, k2, overlap_df
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMBINE BOTH LOG FILES
# ─────────────────────────────────────────────────────────────────────────────

section("COMBINED DATASET — CONCAT")

df = pd.concat([log1, log2], ignore_index=True)
del log1, log2
gc.collect()

print(f"  Total rows      : {len(df):,}")
print(f"  Unique users    : {df['user_id'].nunique():,}")
print(f"  Unique videos   : {df['video_id'].nunique():,}")
print(f"  date range      : {df['date'].min()} – {df['date'].max()}")
print(f"  Memory          : {_rss_mb():.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TAB DISTRIBUTION (full combined dataset)
# ─────────────────────────────────────────────────────────────────────────────

section("TAB DISTRIBUTION — COMBINED")
print("  (tab==1 = main recommendation page / single-column feed)")

total_rows = len(df)
tab_counts = df["tab"].value_counts().sort_index()
print(f"\n  {'tab':>4}  {'count':>12}  {'fraction':>9}")
print(f"  {'─'*30}")
for tv, cnt in tab_counts.items():
    print(f"  {tv:>4}  {cnt:>12,}  {cnt/total_rows:>9.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  DEFINE DATASET A AND DATASET B
# ─────────────────────────────────────────────────────────────────────────────

section("DATASET DEFINITIONS")

dsB = df[df["tab"] == 1].copy()         # Dataset B: all tab==1
dsA = dsB[dsB["is_click"] == 1].copy()  # Dataset A: tab==1 AND valid_play

print(f"  Dataset B (tab==1):              "
      f"{len(dsB):>10,} events | "
      f"{dsB['user_id'].nunique():>5,} users | "
      f"{dsB['video_id'].nunique():>7,} videos")
print(f"  Dataset A (tab==1, is_click==1): "
      f"{len(dsA):>10,} events | "
      f"{dsA['user_id'].nunique():>5,} users | "
      f"{dsA['video_id'].nunique():>7,} videos")
print(f"  Dataset A as fraction of B      : {len(dsA)/len(dsB):.4f}")
print(f"\n  NOTE: is_click==1 on tab==1 means *valid_play* (engagement threshold),")
print(f"  not a screen tap. See KuaiRand README §1.")

# Free the full combined df — we only need dsA and dsB from here.
del df
gc.collect()
print(f"  Memory after freeing full df    : {_rss_mb():.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  IS_CLICK DISTRIBUTION BY TAB (before filtering, use original df logic)
# ─────────────────────────────────────────────────────────────────────────────
# We already dropped df; reconstruct click-rate-by-tab from dsB for tab==1
# and separately note what we can say.

section("IS_CLICK DISTRIBUTION — DATASET B (tab==1)")
n_dsB     = len(dsB)
n_clicked = (dsB["is_click"] == 1).sum()
n_noclick = n_dsB - n_clicked
print(f"  is_click == 1 (valid_play) : {n_clicked:,}  ({n_clicked/n_dsB:.5f})")
print(f"  is_click == 0 (not played) : {n_noclick:,}  ({n_noclick/n_dsB:.5f})")
print(f"  These form Dataset A and the excluded-from-A portion of Dataset B.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. ENGAGEMENT SIGNAL DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def engagement_report(ds: pd.DataFrame, name: str) -> None:
    section(f"ENGAGEMENT SIGNALS — {name}")
    n = len(ds)
    print(f"  Events: {n:,}")

    # Watch ratio
    dur_pos = ds["duration_ms"] > 0
    dur_zero = (ds["duration_ms"] == 0).sum()
    dur_neg  = (ds["duration_ms"] <  0).sum()

    wr_raw = np.where(
        dur_pos,
        ds["play_time_ms"].values / ds["duration_ms"].values.astype(float),
        np.nan,
    )
    wr_series = pd.Series(wr_raw, index=ds.index)
    wr_valid  = wr_series.dropna()
    wr_clipped = wr_valid.clip(0.0, 1.0)

    print(f"\n  Watch ratio (play_time_ms / duration_ms, clipped to [0,1]):")
    print(f"    duration_ms > 0  : {dur_pos.sum():>10,} ({dur_pos.sum()/n:.4f})")
    print(f"    duration_ms == 0 : {dur_zero:>10,} ({dur_zero/n:.4f})  -> watch ratio set to NaN")
    print(f"    duration_ms < 0  : {dur_neg:>10,} ({dur_neg/n:.4f})  -> watch ratio set to NaN")
    print(f"    Unclipped > 1.0  : {(wr_valid > 1.0).sum():>10,} ({(wr_valid > 1.0).mean():.4f})")

    pct_labels = [0.05, 0.25, 0.50, 0.75, 0.95]
    pcts = wr_clipped.describe(percentiles=pct_labels)
    for lbl, val in pcts.items():
        print(f"    {lbl:>10}: {val:.4f}")

    # Binary engagement rates
    print(f"\n  Binary feedback rates (mean over all {n:,} events in {name}):")
    for col in ["is_click", "long_view", "is_like", "is_forward", "is_hate"]:
        rate  = ds[col].mean()
        count = int(ds[col].sum())
        print(f"    {col:<15}: rate={rate:.5f}  count={count:,}")

engagement_report(dsA, "Dataset A (tab==1, is_click==1)")
engagement_report(dsB, "Dataset B (tab==1, all)")

# ─────────────────────────────────────────────────────────────────────────────
# 11. ATTACH AUTHOR METADATA
# ─────────────────────────────────────────────────────────────────────────────

section("AUTHOR METADATA — ATTACH")

def attach_authors(ds: pd.DataFrame, label: str) -> pd.DataFrame:
    out = ds.merge(author_map, on="video_id", how="left")
    n   = len(out)
    n_w = out["author_id"].notna().sum()
    n_x = out["author_id"].isna().sum()
    print(f"  {label}:")
    print(f"    Events with    author_id : {n_w:,}  ({n_w/n:.5f})")
    print(f"    Events missing author_id : {n_x:,}  ({n_x/n:.5f})")
    return out

dsA = attach_authors(dsA, "Dataset A")
dsB = attach_authors(dsB, "Dataset B")
gc.collect()
print(f"  Memory after author join: {_rss_mb():.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# 12. DAILY EVENT COUNTS
# ─────────────────────────────────────────────────────────────────────────────

section("DAILY EVENT COUNTS")
print("  Using the 'date' column directly (YYYYMMDD in UTC+8 / Beijing time).")
print("  This is more reliable than converting time_ms because the raw 'date'")
print("  field in the CSV already reflects the platform's local timezone.")
print("  Assumption: all records are in the same timezone as the 'date' field.\n")

def daily_report(ds: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"  ── {label} ──")
    d_ev   = ds.groupby("date").size().rename("events")
    d_usr  = ds.groupby("date")["user_id"].nunique().rename("users")
    d_vid  = ds.groupby("date")["video_id"].nunique().rename("videos")
    table  = pd.concat([d_ev, d_usr, d_vid], axis=1)
    print(table.to_string())
    print(f"  Total days : {len(table)}")
    print(f"  Events/day : min={d_ev.min():,}  max={d_ev.max():,}  "
          f"median={d_ev.median():.0f}")
    print()
    return table

daily_A = daily_report(dsA, "Dataset A")
daily_B = daily_report(dsB, "Dataset B")

# ─────────────────────────────────────────────────────────────────────────────
# 13. TIMESTAMP TIE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def tie_structure(ds: pd.DataFrame, label: str) -> pd.DataFrame:
    section(f"TIMESTAMP TIE STRUCTURE — {label}")
    print("  Grouping by (user_id, time_ms) to identify ambiguous states.")
    print("  Video-level ambiguity  : group has > 1 event.")
    print("  Author-level validity  : group has exactly 1 distinct non-null author.")

    grp = (
        ds.groupby(["user_id", "time_ms"], sort=True)
        .agg(
            n_events         = ("video_id",   "size"),
            n_distinct_auth  = ("author_id",  "nunique"),   # NaN excluded by nunique
            any_null_auth    = ("author_id",  lambda x: x.isna().any()),
        )
        .reset_index()
    )

    total_grp  = len(grp)
    single_grp = (grp["n_events"] == 1).sum()
    multi_grp  = (grp["n_events"] >  1).sum()
    events_in_multi = grp.loc[grp["n_events"] > 1, "n_events"].sum()

    print(f"\n  Total (user_id, time_ms) groups : {total_grp:,}")
    print(f"  Singleton groups (n_events==1)  : {single_grp:,}  "
          f"({single_grp/total_grp:.4f})")
    print(f"  Ambiguous groups (n_events > 1) : {multi_grp:,}  "
          f"({multi_grp/total_grp:.4f})")
    print(f"  Events inside ambiguous groups  : {events_in_multi:,}")

    # Among multi-event groups: what fraction are valid author states?
    if multi_grp > 0:
        multi = grp[grp["n_events"] > 1].copy()

        # A valid author state: exactly 1 non-null author, no null authors
        valid_auth  = (
            (multi["n_distinct_auth"] == 1) &
            (~multi["any_null_auth"])
        ).sum()
        ambig_auth  = multi_grp - valid_auth

        print(f"\n  Among {multi_grp:,} ambiguous-video groups:")
        print(f"    Valid author states (same single known author) : "
              f"{valid_auth:,}  ({valid_auth/multi_grp:.4f})")
        print(f"    Still ambiguous at author level               : "
              f"{ambig_auth:,}  ({ambig_auth/multi_grp:.4f})")

    # Distribution of group sizes
    print(f"\n  Group size distribution (n_events per timestamp group):")
    gs = grp["n_events"].value_counts().sort_index()
    for sz, cnt in gs.head(10).items():
        bar = "█" * min(int(cnt / gs.max() * 30), 30)
        print(f"    n={sz:>4}: {cnt:>8,}  {bar}")
    if len(gs) > 10:
        print(f"    ... ({len(gs) - 10} more size buckets)")

    return grp

grp_A = tie_structure(dsA, "Dataset A")
grp_B = tie_structure(dsB, "Dataset B")

gc.collect()
print(f"\n  Memory: {_rss_mb():.0f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# 14. CONSERVATIVE TRANSITION COUNTING
# ─────────────────────────────────────────────────────────────────────────────

def conservative_transitions(
    ds: pd.DataFrame,
    ds_label: str,
    gap_label: str,
    gap_ms: int,
) -> dict:
    """
    Count conservative video-level and author-level transitions.

    RULES (Hard Rules 1-3):
    ──────────────────────
    Video-level transition:
      Both source and target timestamp groups must be singletons (n_events==1).
      Ambiguous groups (n_events > 1) are sequence BOUNDARIES; we never
      connect across them.

    Author-level transition:
      A group is a valid author state iff:
        - Exactly 1 distinct non-null author_id in the group
        - No missing author_ids in the group
      Multiple simultaneous videos from one author -> still 1 valid author state.
      Mixed authors or any NaN author -> boundary; no transition.

    Gap measurement:
      Between adjacent group timestamps within the same user.
      Negative gaps cannot occur (groups are grouped by time_ms so consecutive
      group timestamps are strictly increasing within a user).
      A gap exceeding session_gap_ms or occurring at user start (NaN) is
      excluded.

    Transitions assigned to splits: NOT done here.
    This function counts over the full timeline (no split).
    """

    # Build group-level aggregation
    g = (
        ds.groupby(["user_id", "time_ms"], sort=True)
        .agg(
            n_events        = ("video_id",   "size"),
            video_id        = ("video_id",   "first"),   # valid only when n_events==1
            author_id       = ("author_id",  "first"),   # valid only when author_valid==True
            n_distinct_auth = ("author_id",  "nunique"), # NaN excluded
            any_null_auth   = ("author_id",  lambda x: x.isna().any()),
        )
        .reset_index()
        .sort_values(["user_id", "time_ms"])
    )

    # Validity flags
    g["video_valid"]  = g["n_events"] == 1
    g["author_valid"] = (
        (g["n_distinct_auth"] == 1) &
        (~g["any_null_auth"])
    )

    # Shift within user to get previous group fields
    gu = g.groupby("user_id", sort=False)
    g["prev_time"]         = gu["time_ms"].shift(1)
    g["prev_video_id"]     = gu["video_id"].shift(1)
    g["prev_author_id"]    = gu["author_id"].shift(1)
    g["prev_video_valid"]  = gu["video_valid"].shift(1)
    g["prev_author_valid"] = gu["author_valid"].shift(1)

    g["gap_ms"] = g["time_ms"] - g["prev_time"]

    # Within-session filter (gap_ms is NaN for first row of each user -> excluded)
    in_session = (g["gap_ms"] > 0) & (g["gap_ms"] <= gap_ms)

    # ── Video transitions ──────────────────────────────────────────────────
    vid = g[
        in_session &
        g["video_valid"] &
        g["prev_video_valid"].fillna(False)
    ]

    # ── Author transitions ─────────────────────────────────────────────────
    auth = g[
        in_session &
        g["author_valid"] &
        g["prev_author_valid"].fillna(False)
    ]

    # Verify no gap==0 transitions (should be impossible by construction)
    assert (vid["gap_ms"] > 0).all(),  "BUG: gap==0 video transition found"
    assert (auth["gap_ms"] > 0).all(), "BUG: gap==0 author transition found"

    n_vid  = len(vid)
    n_auth = len(auth)

    r = {
        "dataset"          : ds_label,
        "gap"              : gap_label,
        "video_transitions": n_vid,
        "video_users"      : vid["user_id"].nunique() if n_vid else 0,
        "video_src_unique" : vid["prev_video_id"].nunique() if n_vid else 0,
        "video_tgt_unique" : vid["video_id"].nunique()      if n_vid else 0,
        "auth_transitions" : n_auth,
        "auth_users"       : auth["user_id"].nunique()       if n_auth else 0,
        "auth_src_unique"  : auth["prev_author_id"].nunique() if n_auth else 0,
        "auth_tgt_unique"  : auth["author_id"].nunique()      if n_auth else 0,
    }

    print(f"\n  [{ds_label:20s} | gap={gap_label}]")
    print(f"    Video  transitions : {n_vid:>10,}   users={r['video_users']:,}  "
          f"src_states={r['video_src_unique']:,}  tgt_states={r['video_tgt_unique']:,}")
    print(f"    Author transitions : {n_auth:>10,}   users={r['auth_users']:,}  "
          f"src_states={r['auth_src_unique']:,}  tgt_states={r['auth_tgt_unique']:,}")

    return r

section("CONSERVATIVE TRANSITION COUNTS  (full timeline — diagnostic only)")
print("  These counts do NOT represent training/validation/test splits.")
print("  They are conservative (ambiguous groups = boundaries).")
print("  No transition probabilities are estimated or stored.\n")

results = []
for gap_lbl, gap_ms in SESSION_GAPS.items():
    results.append(
        conservative_transitions(dsA, "Dataset A", gap_lbl, gap_ms)
    )
    results.append(
        conservative_transitions(dsB, "Dataset B", gap_lbl, gap_ms)
    )

# ─────────────────────────────────────────────────────────────────────────────
# 15. TRANSITION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

section("TRANSITION SUMMARY TABLE")

tr_df = pd.DataFrame(results)
print(tr_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 16. SOURCE STATE REPEAT FREQUENCY  (30-min, Dataset A, full timeline)
# ─────────────────────────────────────────────────────────────────────────────

section("SOURCE STATE REPEAT FREQUENCY — Dataset A, 30-min gap (full timeline)")
print("  This tells us how often each source state has been observed, which")
print("  directly affects transition probability estimation reliability.")

gap_ms_30 = SESSION_GAPS["30min"]

g_A = (
    dsA.groupby(["user_id", "time_ms"], sort=True)
    .agg(
        n_events        = ("video_id",  "size"),
        video_id        = ("video_id",  "first"),
        author_id       = ("author_id", "first"),
        n_distinct_auth = ("author_id", "nunique"),
        any_null_auth   = ("author_id", lambda x: x.isna().any()),
    )
    .reset_index()
    .sort_values(["user_id", "time_ms"])
)
g_A["video_valid"]  = g_A["n_events"] == 1
g_A["author_valid"] = (g_A["n_distinct_auth"] == 1) & (~g_A["any_null_auth"])

gu_A = g_A.groupby("user_id", sort=False)
g_A["prev_video_id"]     = gu_A["video_id"].shift(1)
g_A["prev_author_id"]    = gu_A["author_id"].shift(1)
g_A["prev_video_valid"]  = gu_A["video_valid"].shift(1)
g_A["prev_author_valid"] = gu_A["author_valid"].shift(1)
g_A["gap_ms"]            = g_A["time_ms"] - gu_A["time_ms"].shift(1)

in_sess = (g_A["gap_ms"] > 0) & (g_A["gap_ms"] <= gap_ms_30)

vid_30  = g_A[in_sess & g_A["video_valid"] & g_A["prev_video_valid"].fillna(False)]
auth_30 = g_A[in_sess & g_A["author_valid"] & g_A["prev_author_valid"].fillna(False)]

for name, df_t, col in [
    ("Video  source states", vid_30,  "prev_video_id"),
    ("Author source states", auth_30, "prev_author_id"),
]:
    freq = df_t.groupby(col).size()
    print(f"\n  {name}  ({len(freq):,} unique):")
    desc = freq.describe(percentiles=[.25, .50, .75, .90, .95])
    for lbl, val in desc.items():
        print(f"    {lbl:>6}: {val:,.1f}")
    print(f"    Observed once     : {(freq == 1).sum():,}  ({(freq==1).mean():.4f})")
    print(f"    Observed 2–4 times: {((freq>=2)&(freq<=4)).sum():,}")
    print(f"    Observed 5+ times : {(freq >= 5).sum():,}")

del g_A, vid_30, auth_30
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 17. FINAL MEMORY AND TIMING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("PROFILING COMPLETE")
print(f"  Total elapsed time : {_elapsed()}")
print(f"  Final memory RSS   : {_rss_mb():.0f} MB")
print()
print("  ASSUMPTIONS THAT STILL NEED VERIFICATION:")
print("  1. The 'date' column faithfully reflects UTC+8 for ALL records.")
print("     Verify: check whether time_ms -> UTC+8 matches the 'date' field.")
print("  2. No systematic batch-ingestion artefacts create false timestamp ties.")
print("     Verify: inspect the tie-size distribution.  Extreme spikes are suspicious.")
print("  3. video_features_basic_1k.csv is the authoritative author mapping.")
print("     If a video appears in the logs but not in this file, author is unknown.")
print("  4. is_rand events in the standard log files are genuinely random exposures.")
print("     Verify: check is_rand distribution within tab==1 to decide if they")
print("     should be included or excluded from sequence construction.")
print("  5. The two log files cover non-overlapping calendar periods.")
print("     Verify: cross-file date-overlap check output above.")
print("  6. 30-min and 60-min gap thresholds are reasonable for this platform.")
print("     Verify: examine the gap distribution from the transition counts.")
print()
print("  NEXT STEPS  (require your review of actual output before proceeding):")
print("  1. Review the daily event count tables to choose calendar-based split dates.")
print("  2. Confirm the is_click / valid_play interpretation matches study design.")
print("  3. Decide whether to include is_rand==1 events in the study sequences.")
print("  4. Confirm which session gap threshold to use as primary.")
print()
print("  NO ARTEFACTS WRITTEN.  NO FILTERING PERFORMED.  NO MODEL TRAINED.")
