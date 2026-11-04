
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = (
    r"D:\APPS Research\KuaiRand-1K\data"
    r"\kuairand_tab1_5core.csv"
)

VIDEO_FILE = (
    r"D:\APPS Research\KuaiRand-1K\data"
    r"\video_features_basic_1k.csv"
)

SESSION_GAP_MS = 30 * 60 * 1000

print("Loading interaction data...")
df = pd.read_csv(DATA_FILE).reset_index(drop=True)

# Preserve original CSV row order for diagnostics only.
df["original_row"] = np.arange(len(df))

print("Loading video metadata...")
vf = pd.read_csv(VIDEO_FILE)

print("\nMetadata columns:")
print(vf.columns.tolist())

if "author_id" not in vf.columns or "video_id" not in vf.columns:
    raise ValueError("video_id or author_id is missing from metadata.")

# ============================================================
# 1. VALIDATE VIDEO-AUTHOR MAPPING
# ============================================================

mapping = vf[["video_id", "author_id"]].dropna(
    subset=["video_id", "author_id"]
).copy()

author_counts = mapping.groupby("video_id")["author_id"].nunique()

ambiguous_videos = set(
    author_counts[author_counts > 1].index
)

print("\n========== AUTHOR MAPPING ==========")
print("Metadata videos:", vf["video_id"].nunique())
print("Unique authors:", vf["author_id"].nunique())
print("Videos with conflicting author IDs:", len(ambiguous_videos))

# Keep only unambiguous video-to-author mappings.
mapping = mapping[
    ~mapping["video_id"].isin(ambiguous_videos)
].drop_duplicates("video_id")

df = df.merge(
    mapping,
    on="video_id",
    how="left",
    validate="many_to_one"
)

print(
    "Events with author:",
    round(df["author_id"].notna().mean(), 4)
)

# ============================================================
# 2. GLOBAL CHRONOLOGICAL SPLIT
# ============================================================

df = df.sort_values(
    ["user_id", "time_ms", "original_row"],
    kind="mergesort"
).copy()

t1, t2 = df["time_ms"].quantile([0.70, 0.85])

print("\n========== TIME SPLIT ==========")
print("Training cutoff:", t1)
print("Validation cutoff:", t2)

# ============================================================
# 3. NAIVE ROW-ORDER TRANSITIONS
#    Diagnostic comparison only, not ground truth.
# ============================================================

def naive_transitions(data, column):

    d = data.sort_values(
        ["user_id", "time_ms", "original_row"],
        kind="mergesort"
    ).copy()

    grouped = d.groupby("user_id", sort=False)

    d["src"] = grouped[column].shift(1)
    d["previous_time"] = grouped["time_ms"].shift(1)

    d["gap"] = d["time_ms"] - d["previous_time"]

    result = d[
        d["src"].notna() &
        d[column].notna() &
        (d["gap"] >= 0) &
        (d["gap"] <= SESSION_GAP_MS)
    ].copy()

    result["tgt"] = result[column]

    return result

# ============================================================
# 4. CONSERVATIVE TIMESTAMP-GROUP TRANSITIONS
# ============================================================

groups = (
    df.groupby(["user_id", "time_ms"], sort=False)
    .agg(
        event_count=("video_id", "size"),
        video_id=("video_id", "first"),
        author_id=("author_id", "first"),
        author_count=("author_id", "nunique")
    )
    .reset_index()
    .sort_values(["user_id", "time_ms"], kind="mergesort")
)

grouped = groups.groupby("user_id", sort=False)

groups["previous_time"] = grouped["time_ms"].shift(1)
groups["previous_video"] = grouped["video_id"].shift(1)
groups["previous_author"] = grouped["author_id"].shift(1)

groups["previous_event_count"] = (
    grouped["event_count"].shift(1)
)

groups["previous_author_count"] = (
    grouped["author_count"].shift(1)
)

groups["gap"] = groups["time_ms"] - groups["previous_time"]

# Video transitions require both timestamp groups to be singletons.
video_pairs = groups[
    (groups["event_count"] == 1) &
    (groups["previous_event_count"] == 1) &
    (groups["gap"] >= 0) &
    (groups["gap"] <= SESSION_GAP_MS)
].copy()

video_pairs["src"] = video_pairs["previous_video"]
video_pairs["tgt"] = video_pairs["video_id"]

# Author transitions:
# Multiple simultaneous videos from the same author still represent
# one unambiguous author state. Multiple authors at one timestamp
# remain ambiguous.
author_pairs = groups[
    groups["author_id"].notna() &
    groups["previous_author"].notna() &
    (groups["author_count"] == 1) &
    (groups["previous_author_count"] == 1) &
    (groups["gap"] >= 0) &
    (groups["gap"] <= SESSION_GAP_MS)
].copy()

author_pairs["src"] = author_pairs["previous_author"]
author_pairs["tgt"] = author_pairs["author_id"]

# ============================================================
# 5. ASSIGN TRANSITIONS BY TARGET TIME
# ============================================================

def split_transitions(pairs):

    train = pairs[pairs["time_ms"] <= t1].copy()

    val = pairs[
        (pairs["time_ms"] > t1) &
        (pairs["time_ms"] <= t2)
    ].copy()

    test = pairs[pairs["time_ms"] > t2].copy()

    return train, val, test

# Full timeline is used to preserve cross-boundary history.
video_train, video_val, video_test = split_transitions(video_pairs)
author_train, author_val, author_test = split_transitions(author_pairs)

# ============================================================
# 6. REPORT TRANSITION COUNTS
# ============================================================

print("\n========== TRANSITION COUNTS ==========")

print("Training events:", (df["time_ms"] <= t1).sum())
print("Validation events:",
      ((df["time_ms"] > t1) & (df["time_ms"] <= t2)).sum())
print("Testing events:", (df["time_ms"] > t2).sum())

print("\nNaive row-order video transitions:")
print("Train:", len(naive_transitions(df[df["time_ms"] <= t1], "video_id")))
print("Test:", len(naive_transitions(df[df["time_ms"] > t2], "video_id")))

print("\nConservative video transitions:")
print("Train:", len(video_train))
print("Validation:", len(video_val))
print("Test:", len(video_test))

print("\nConservative author transitions:")
print("Train:", len(author_train))
print("Validation:", len(author_val))
print("Test:", len(author_test))

# ============================================================
# 7. VIDEO-LEVEL COVERAGE
# ============================================================

video_catalogue = set(
    df.loc[df["time_ms"] <= t1, "video_id"]
)

video_sources = set(video_train["src"])

print("\n========== VIDEO-LEVEL COVERAGE ==========")

if len(video_test):
    print(
        "Test targets known:",
        round(video_test["tgt"].isin(video_catalogue).mean(), 4)
    )

    print(
        "Test sources known:",
        round(video_test["src"].isin(video_sources).mean(), 4)
    )

# ============================================================
# 8. AUTHOR-LEVEL COVERAGE
# ============================================================

author_catalogue = set(
    df.loc[
        (df["time_ms"] <= t1) &
        df["author_id"].notna(),
        "author_id"
    ]
)

author_sources = set(author_train["src"])

print("\n========== AUTHOR-LEVEL COVERAGE ==========")

print("Training authors:", len(author_catalogue))
print("Training author source states:", len(author_sources))

if len(author_test):

    print(
        "Test targets known:",
        round(author_test["tgt"].isin(author_catalogue).mean(), 4)
    )

    print(
        "Test sources known:",
        round(author_test["src"].isin(author_sources).mean(), 4)
    )

# ============================================================
# 9. SOURCE REPEAT FREQUENCY
# ============================================================

print("\n========== SOURCE REPEAT FREQUENCY ==========")

for name, transitions in [
    ("Video", video_train),
    ("Author", author_train)
]:

    frequencies = transitions.groupby("src").size()

    print(f"\n{name} states:")

    if len(frequencies):
        print(frequencies.describe())
        print("Sources appearing once:", (frequencies == 1).sum())
        print("Sources appearing 5+ times:", (frequencies >= 5).sum())
    else:
        print("No valid transitions.")

# ============================================================
# 10. SAVE AUTHOR TRANSITIONS
# ============================================================

OUTPUT_DIR = r"D:\APPS Research\KuaiRand-1K\data"

author_train.to_csv(
    OUTPUT_DIR + r"\author_train_transitions.csv",
    index=False
)

author_val.to_csv(
    OUTPUT_DIR + r"\author_val_transitions.csv",
    index=False
)

author_test.to_csv(
    OUTPUT_DIR + r"\author_test_transitions.csv",
    index=False
)

print("\n========== DIAGNOSTIC COMPLETE ==========")
