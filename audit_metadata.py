"""
READ-ONLY metadata audit for KuaiRand-1K (stage H0/H1).

- Does NOT modify any Phase 0-3 file. Writes only to ./audit_output/
- Does NOT read the standard log that contains the test period.
  It reads only log_standard_4_08_to_4_21_1k.csv (8-21 April, training period).
- Does NOT assume column names: it prints what is actually in the files,
  then profiles any column whose name contains 'author' or 'tag' or 'categ'.

Usage (Windows, from any folder):
    python audit_metadata.py "D:\\APPS Research\\KuaiRand-1K\\data"
"""
import os
import sys
import glob
import json
import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR = "audit_output"
os.makedirs(OUT_DIR, exist_ok=True)
report = {}


def log(msg=""):
    print(msg)


# ---------------------------------------------------------------- 1. files
log("=== 1. Files and headers (first 3 rows only) ===")
for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
    try:
        head = pd.read_csv(path, nrows=3)
        log(f"\n{os.path.basename(path)}  ({os.path.getsize(path)/1e6:.0f} MB)")
        log("  columns: " + ", ".join(head.columns))
        report[os.path.basename(path)] = list(head.columns)
    except Exception as e:  # keep going, report the problem
        log(f"  could not read: {e}")

# -------------------------------------------------- 2. video basic features
basic_candidates = glob.glob(os.path.join(DATA_DIR, "video_features_basic*1k*.csv"))
if not basic_candidates:
    log("\nNo video_features_basic*1k*.csv found; stopping.")
    sys.exit(0)

basic_path = basic_candidates[0]
log(f"\n=== 2. Profiling {os.path.basename(basic_path)} ===")
vf = pd.read_csv(basic_path, low_memory=False)
log(f"rows: {len(vf):,}   distinct video_id: {vf['video_id'].nunique():,}"
    if "video_id" in vf.columns else f"rows: {len(vf):,}")

profile = []
for col in vf.columns:
    s = vf[col]
    top = s.value_counts(dropna=False).head(5)
    profile.append({
        "column": col,
        "non_null_pct": round(100 * s.notna().mean(), 3),
        "n_unique": int(s.nunique(dropna=True)),
        # most frequent values reveal placeholder/sentinel codes
        "top5_values": {str(k): int(v) for k, v in top.items()},
    })
pd.DataFrame(profile).to_csv(os.path.join(OUT_DIR, "video_features_column_profile.csv"),
                             index=False)
log("column profile saved to audit_output/video_features_column_profile.csv")

# ------------------------------------------- 3. author / tag / category cols
keys = [c for c in vf.columns if any(k in c.lower() for k in ("author", "tag", "categ"))]
log(f"\n=== 3. Candidate hierarchy columns: {keys} ===")
for col in keys:
    s = vf[col]
    log(f"\n-- {col} --")
    log(f"non-null: {100*s.notna().mean():.2f}%   unique: {s.nunique():,}")
    log("top 10 values (look for sentinel/placeholder codes):")
    log(s.value_counts(dropna=False).head(10).to_string())
    if s.dtype == object:
        # multi-valued check: do cells contain a separator?
        sample = s.dropna().astype(str).head(200000)
        has_sep = sample.str.contains(",").mean()
        log(f"share of cells containing a comma (multi-valued?): {100*has_sep:.2f}%")
        if has_sep > 0:
            lens = sample.str.split(",").map(len)
            log("list-length distribution:")
            log(lens.value_counts().sort_index().head(10).to_string())
    # group-size distribution (videos per group)
    sizes = s.value_counts(dropna=True)
    log("videos per group: median=%d  mean=%.1f  max=%d  groups with 1 video=%.1f%%" % (
        sizes.median(), sizes.mean(), sizes.max(), 100 * (sizes == 1).mean()))

# --------------- 4. coverage over TRAINING-ONLY log (8-21 April file only)
train_log = glob.glob(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_1k.csv"))
if train_log and "video_id" in vf.columns:
    log("\n=== 4. Metadata coverage of videos in the 8-21 April log ===")
    vids = pd.read_csv(train_log[0], usecols=["video_id"])["video_id"].drop_duplicates()
    log(f"distinct videos in this log: {len(vids):,}")
    merged = vids.to_frame().merge(vf, on="video_id", how="left", indicator=True)
    log(f"found in video_features_basic: {100*(merged['_merge']=='both').mean():.2f}%")
    for col in keys:
        log(f"  {col}: non-null for {100*merged[col].notna().mean():.2f}% of these videos")
else:
    log("\n(4 skipped: training log or video_id column not found)")

with open(os.path.join(OUT_DIR, "headers.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
log("\nDone. Paste the printed output (or the audit_output folder files) back to me.")
