"""
kr_pipeline.py  —  Leak-Free Chronological Transition Pipeline
==============================================================

PURPOSE
  Extracts conservative sequence transitions and partitions them into
  leak-free chronological splits: Train, Buffer, Validation, and Test.
  
HARD RULES ENFORCED
  1. No 5-core filtering.
  2. No CSV row order used for simultaneous events (identical user_id, time_ms).
  3. Ambiguous timestamp groups remain boundaries; no bridging transitions across them.
  4. Each transition (src -> target) is assigned to a split based on the
     TARGET event's timestamp.
  5. Training candidate catalogue and popularity statistics derive strictly from TRAIN.
  6. Author mapping validated; raises an error on any conflicting video -> author mapping.
  7. Buffer period (April 29) events are preserved as valid previous-event context
     for validation transitions, but transitions targeting April 29 are excluded
     from training, validation, and testing.

OUTPUT ARTEFACTS
  data/splits/
    ├── train_transitions_30min.parquet
    ├── val_transitions_30min.parquet
    ├── test_transitions_30min.parquet
    ├── train_catalogue_30min.json
    ├── author_video_map_30min.json
    ├── video_popularity_30min.json
    ├── author_popularity_30min.json
    └── split_config_30min.json
"""

import gc
import json
import sys
import time
from pathlib import Path
import psutil

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(r"D:\APPS Research\KuaiRand-1K\data")
OUTPUT_DIR = DATA_DIR / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_1 = DATA_DIR / "log_standard_4_08_to_4_21_1k.csv"
LOG_FILE_2 = DATA_DIR / "log_standard_4_22_to_5_08_1k.csv"
VIDEO_META = DATA_DIR / "video_features_basic_1k.csv"

# Exact millisecond boundaries (aligned with UTC+8 / China Standard Time)
SPLIT_BOUNDARIES_MS = {
    "train_start": 1649347200000,  # 2022-04-08 00:00:00 CST
    "train_end":   1651161599999,  # 2022-04-28 23:59:59.999 CST
    "buffer_start":1651161600000,  # 2022-04-29 00:00:00 CST
    "buffer_end":  1651247999999,  # 2022-04-29 23:59:59.999 CST
    "val_start":   1651248000000,  # 2022-04-30 00:00:00 CST
    "val_end":     1651593599999,  # 2022-05-03 23:59:59.999 CST
    "test_start":  1651593600000,  # 2022-05-04 00:00:00 CST
    "test_end":    1652025599999,  # 2022-05-08 23:59:59.999 CST
}

LOG_COLS = ["user_id", "video_id", "time_ms", "tab", "is_click"]
LOG_DTYPES = {
    "user_id":  "int32",
    "video_id": "int64",
    "time_ms":  "int64",
    "tab":      "int8",
    "is_click": "int8",
}

def log_mem():
    mb = psutil.Process().memory_info().rss / 1_048_576
    return f"{mb:.0f} MB"

def build_pipeline(session_gap_minutes=30, dataset_variant="DatasetA"):
    t0 = time.time()
    session_gap_ms = session_gap_minutes * 60 * 1000
    tag = f"{session_gap_minutes}min"
    print(f"\n{'='*70}")
    print(f"STARTING PIPELINE: {dataset_variant} | Session Gap: {session_gap_minutes} min")
    print(f"Memory: {log_mem()}")
    print(f"{'='*70}\n")

    # 1. Load video metadata and validate author mapping
    print("1. Loading video metadata and validating author mapping...")
    vf = pd.read_csv(VIDEO_META, usecols=["video_id", "author_id"], dtype={"video_id": "int64"})
    vf["author_id"] = pd.to_numeric(vf["author_id"], errors="coerce")
    
    # Conflict check (Hard Rule 6)
    conflicts = (
        vf.dropna(subset=["author_id"])
        .groupby("video_id")["author_id"]
        .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if len(conflicts) > 0:
        raise ValueError(f"HARD RULE 6 VIOLATION: {len(conflicts)} videos have conflicting authors!")
    
    author_map = (
        vf.dropna(subset=["author_id"])
        .drop_duplicates(subset=["video_id"], keep="first")
        [["video_id", "author_id"]]
        .copy()
    )
    author_map["author_id"] = author_map["author_id"].astype("int64")
    del vf
    gc.collect()
    print(f"  Metadata verified: {len(author_map):,} unique video->author mappings. Mem: {log_mem()}")

    # 2. Load log files
    print("\n2. Loading interaction logs...")
    log1 = pd.read_csv(LOG_FILE_1, usecols=LOG_COLS, dtype=LOG_DTYPES)
    log2 = pd.read_csv(LOG_FILE_2, usecols=LOG_COLS, dtype=LOG_DTYPES)
    df = pd.concat([log1, log2], ignore_index=True)
    del log1, log2
    gc.collect()

    print(f"  Total raw events loaded: {len(df):,}. Mem: {log_mem()}")

    # 3. Filter Dataset
    if dataset_variant == "DatasetA":
        # tab == 1 AND valid_play (is_click == 1)
        df = df[(df["tab"] == 1) & (df["is_click"] == 1)].copy()
    elif dataset_variant == "DatasetB":
        df = df[df["tab"] == 1].copy()
    else:
        raise ValueError(f"Unknown dataset_variant: {dataset_variant}")

    df = df[["user_id", "video_id", "time_ms"]].copy()
    print(f"  Filtered events for {dataset_variant}: {len(df):,}. Mem: {log_mem()}")

    # 4. Attach author metadata
    print("\n4. Attaching author metadata...")
    df = df.merge(author_map, on="video_id", how="left")
    missing_authors = df["author_id"].isna().sum()
    if missing_authors > 0:
        print(f"  WARNING: {missing_authors:,} events missing author_id.")

    # 5. Build conservative timestamp groups
    print("\n5. Aggregating into conservative timestamp groups (user_id, time_ms)...")
    # Grouping by (user_id, time_ms) - DO NOT rely on CSV row order for ties
    grp = (
        df.groupby(["user_id", "time_ms"], sort=True)
        .agg(
            n_events=("video_id", "size"),
            video_id=("video_id", "first"),
            author_id=("author_id", "first"),
            n_distinct_auth=("author_id", "nunique"),
            any_null_auth=("author_id", lambda x: x.isna().any()),
        )
        .reset_index()
        .sort_values(["user_id", "time_ms"])
    )

    grp["video_valid"] = grp["n_events"] == 1
    grp["author_valid"] = (grp["n_distinct_auth"] == 1) & (~grp["any_null_auth"])

    print(f"  Total timestamp groups: {len(grp):,}")
    print(f"  Singleton groups (valid video state): {grp['video_valid'].sum():,} ({grp['video_valid'].mean():.2%})")
    print(f"  Ambiguous groups (boundaries): {(~grp['video_valid']).sum():,}")

    # 6. Extract sequential transitions within user
    print(f"\n6. Extracting transitions within {session_gap_minutes}-min session gaps...")
    gu = grp.groupby("user_id", sort=False)
    grp["prev_time"]         = gu["time_ms"].shift(1)
    grp["prev_video_id"]     = gu["video_id"].shift(1)
    grp["prev_author_id"]    = gu["author_id"].shift(1)
    grp["prev_video_valid"]  = gu["video_valid"].shift(1)
    grp["prev_author_valid"] = gu["author_valid"].shift(1)
    grp["gap_ms"]            = grp["time_ms"] - grp["prev_time"]

    # Filter: strictly positive gap, within session limit, both endpoints valid
    in_session = (grp["gap_ms"] > 0) & (grp["gap_ms"] <= session_gap_ms)

    # Video transitions
    v_mask = in_session & grp["video_valid"] & grp["prev_video_valid"].fillna(False)
    transitions = pd.DataFrame({
        "user_id":        grp.loc[v_mask, "user_id"],
        "src_video_id":   grp.loc[v_mask, "prev_video_id"].astype("int64"),
        "tgt_video_id":   grp.loc[v_mask, "video_id"].astype("int64"),
        "src_author_id":  grp.loc[v_mask, "prev_author_id"].astype("int64"),
        "tgt_author_id":  grp.loc[v_mask, "author_id"].astype("int64"),
        "src_time_ms":    grp.loc[v_mask, "prev_time"].astype("int64"),
        "tgt_time_ms":    grp.loc[v_mask, "time_ms"].astype("int64"),
        "gap_ms":         grp.loc[v_mask, "gap_ms"].astype("int64"),
    }).reset_index(drop=True)

    print(f"  Valid conservative transitions created: {len(transitions):,}")

    # Clean up intermediate group frame
    del grp
    gc.collect()

    # 7. Chronological Partitioning by TARGET event timestamp
    print("\n7. Partitioning transitions into chronological splits by target timestamp...")
    t_start_train = SPLIT_BOUNDARIES_MS["train_start"]
    t_end_train   = SPLIT_BOUNDARIES_MS["train_end"]
    t_start_val   = SPLIT_BOUNDARIES_MS["val_start"]
    t_end_val     = SPLIT_BOUNDARIES_MS["val_end"]
    t_start_test  = SPLIT_BOUNDARIES_MS["test_start"]
    t_end_test    = SPLIT_BOUNDARIES_MS["test_end"]

    train_trans = transitions[
        (transitions["tgt_time_ms"] >= t_start_train) &
        (transitions["tgt_time_ms"] <= t_end_train)
    ].copy()

    val_trans = transitions[
        (transitions["tgt_time_ms"] >= t_start_val) &
        (transitions["tgt_time_ms"] <= t_end_val)
    ].copy()

    test_trans = transitions[
        (transitions["tgt_time_ms"] >= t_start_test) &
        (transitions["tgt_time_ms"] <= t_end_test)
    ].copy()

    # Buffer check: transitions targeting the buffer day are excluded
    buffer_trans = transitions[
        (transitions["tgt_time_ms"] >= SPLIT_BOUNDARIES_MS["buffer_start"]) &
        (transitions["tgt_time_ms"] <= SPLIT_BOUNDARIES_MS["buffer_end"])
    ]
    print(f"  Train transitions      : {len(train_trans):,}")
    print(f"  Buffer transitions (cut): {len(buffer_trans):,} (April 29 target events excluded)")
    print(f"  Val transitions        : {len(val_trans):,}")
    print(f"  Test transitions       : {len(test_trans):,}")

    # Check preserved context across boundaries:
    val_cross_ctx = (val_trans["src_time_ms"] < t_start_val).sum()
    test_cross_ctx = (test_trans["src_time_ms"] < t_start_test).sum()
    print(f"  Val transitions with cross-boundary context (source before Apr 30): {val_cross_ctx:,}")
    print(f"  Test transitions with cross-boundary context (source before May 4): {test_cross_ctx:,}")

    # 8. Construct Training-Only Candidate Catalogue and Popularity Tables
    print("\n8. Building training-only candidate catalogue and popularity structures...")
    # Catalogue is derived strictly from training events (all training events up to t_end_train)
    train_events = df[df["time_ms"] <= t_end_train].copy()
    
    # Video training catalogue
    train_catalogue = sorted(int(x) for x in train_events["video_id"].unique())
    catalogue_set = set(train_catalogue)
    print(f"  Training candidate catalogue size: {len(train_catalogue):,} videos")

    # Video popularity (interaction frequency in train)
    vid_pop = train_events["video_id"].value_counts().to_dict()
    # Author popularity (interaction frequency in train)
    auth_pop = train_events["author_id"].dropna().astype("int64").value_counts().to_dict()

    # Author -> video lookup within training catalogue
    print("  Mapping author -> videos within training catalogue...")
    auth_vid_map = {}
    train_author_events = train_events.dropna(subset=["author_id"])[["author_id", "video_id"]].drop_duplicates()
    for row in train_author_events.itertuples(index=False):
        aid = int(row.author_id)
        vid = int(row.video_id)
        if aid not in auth_vid_map:
            auth_vid_map[aid] = []
        auth_vid_map[aid].append(vid)

    # Sort each author's videos by training popularity
    for aid in auth_vid_map:
        auth_vid_map[aid].sort(key=lambda v: vid_pop.get(v, 0), reverse=True)

    print(f"  Unique authors in training catalogue: {len(auth_vid_map):,}")

    # 9. Verify and save artefacts
    print("\n9. Saving dataset artefacts...")
    train_trans.to_parquet(OUTPUT_DIR / f"train_transitions_{tag}.parquet", index=False)
    val_trans.to_parquet(OUTPUT_DIR / f"val_transitions_{tag}.parquet", index=False)
    test_trans.to_parquet(OUTPUT_DIR / f"test_transitions_{tag}.parquet", index=False)

    with open(OUTPUT_DIR / f"train_catalogue_{tag}.json", "w") as f:
        json.dump(train_catalogue, f)

    with open(OUTPUT_DIR / f"video_popularity_{tag}.json", "w") as f:
        json.dump({str(k): int(v) for k, v in vid_pop.items()}, f)

    with open(OUTPUT_DIR / f"author_popularity_{tag}.json", "w") as f:
        json.dump({str(k): int(v) for k, v in auth_pop.items()}, f)

    with open(OUTPUT_DIR / f"author_video_map_{tag}.json", "w") as f:
        json.dump({str(k): v for k, v in auth_vid_map.items()}, f)

    config = {
        "dataset_variant": dataset_variant,
        "session_gap_minutes": session_gap_minutes,
        "split_boundaries_ms": SPLIT_BOUNDARIES_MS,
        "train_transitions": len(train_trans),
        "val_transitions": len(val_trans),
        "test_transitions": len(test_trans),
        "catalogue_size": len(train_catalogue),
        "unique_train_authors": len(auth_vid_map),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_DIR / f"split_config_{tag}.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nPipeline {tag} completed successfully in {elapsed:.1f}s. Final Mem: {log_mem()}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Primary pipeline: 30-min gap
    build_pipeline(session_gap_minutes=30, dataset_variant="DatasetA")
    # Sensitivity pipeline: 60-min gap
    build_pipeline(session_gap_minutes=60, dataset_variant="DatasetA")
