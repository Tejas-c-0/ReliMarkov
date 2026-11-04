
import pandas as pd
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE = r"D:\APPS Research\KuaiRand-1K\data\kuairand_tab1_5core.csv"

SESSION_GAP_MS = 30 * 60 * 1000

print("Loading 5-core dataset...")

df = pd.read_csv(FILE)

required_columns = [
    "user_id",
    "video_id",
    "time_ms",
    "play_time_ms",
    "duration_ms",
    "long_view",
    "is_like",
    "is_forward",
    "is_hate"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["user_id", "video_id", "time_ms"]).copy()

df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
df = df.dropna(subset=["time_ms"]).copy()

print("Interactions:", len(df))
print("Users:", df["user_id"].nunique())
print("Videos:", df["video_id"].nunique())

# ============================================================
# 2. GLOBAL CHRONOLOGICAL SPLIT BOUNDARIES
# ============================================================

t1, t2 = df["time_ms"].quantile([0.70, 0.85])

print("\n========== TIME SPLIT ==========")
print("Training cutoff:", t1)
print("Validation cutoff:", t2)

# ============================================================
# 3. CALCULATE EVENT RELIABILITY
# ============================================================

df["play_time_ms"] = pd.to_numeric(
    df["play_time_ms"], errors="coerce"
).fillna(0)

df["duration_ms"] = pd.to_numeric(
    df["duration_ms"], errors="coerce"
)

for col in ["long_view", "is_like", "is_forward", "is_hate"]:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).fillna(0)

# Duration <= 0: watch ratio is set to zero.
df["watch_ratio"] = np.where(
    df["duration_ms"] > 0,
    df["play_time_ms"] / df["duration_ms"],
    0
)

df["watch_ratio"] = np.clip(
    df["watch_ratio"], 0, 1
)

df["event_weight"] = (
    df["watch_ratio"]
    + 0.3 * df["long_view"]
    + 0.3 * df["is_like"]
    + 0.2 * df["is_forward"]
    - 0.5 * df["is_hate"]
)

df["event_weight"] = df["event_weight"].clip(0.05, 1.5)

# ============================================================
# 4. GROUP EVENTS BY USER AND TIMESTAMP
# ============================================================

df = df.sort_values(
    ["user_id", "time_ms"]
).copy()

groups = (
    df.groupby(["user_id", "time_ms"], sort=False)
    .agg(
        event_count=("video_id", "size"),
        video_id=("video_id", "first"),
        event_weight=("event_weight", "first")
    )
    .reset_index()
    .sort_values(["user_id", "time_ms"])
)

# Preserve ambiguous timestamp groups.
# We do not invent an order between simultaneous videos.

groups["previous_count"] = (
    groups.groupby("user_id")["event_count"].shift(1)
)

groups["previous_video"] = (
    groups.groupby("user_id")["video_id"].shift(1)
)

groups["previous_weight"] = (
    groups.groupby("user_id")["event_weight"].shift(1)
)

groups["previous_time"] = (
    groups.groupby("user_id")["time_ms"].shift(1)
)

groups["gap_ms"] = (
    groups["time_ms"] - groups["previous_time"]
)

# ============================================================
# 5. BUILD TRANSITIONS ON FULL TIMELINE
# ============================================================

pairs = groups[
    (groups["event_count"] == 1) &
    (groups["previous_count"] == 1) &
    (groups["gap_ms"] >= 0) &
    (groups["gap_ms"] <= SESSION_GAP_MS)
].copy()

pairs["src"] = pairs["previous_video"]
pairs["target"] = pairs["video_id"]

pairs["w_src"] = pairs["previous_weight"]
pairs["w_target"] = pairs["event_weight"]

pairs["w_transition"] = np.sqrt(
    pairs["w_src"] * pairs["w_target"]
)

pairs = pairs.rename(columns={
    "user_id": "user_id",
    "time_ms": "target_time_ms"
})

# ============================================================
# 6. ASSIGN TRANSITIONS BY TARGET EVENT TIME
# ============================================================

# Important:
# Build transitions across the full timeline first.
# Assign each transition according to its target timestamp.
# This preserves the previous event across split boundaries.

train = pairs[
    pairs["target_time_ms"] <= t1
].copy()

val = pairs[
    (pairs["target_time_ms"] > t1) &
    (pairs["target_time_ms"] <= t2)
].copy()

test = pairs[
    pairs["target_time_ms"] > t2
].copy()

# ============================================================
# 7. TRAINING-ONLY CATALOGUE AND SOURCE STATES
# ============================================================

train_events = df[
    df["time_ms"] <= t1
].copy()

catalogue = set(train_events["video_id"])

known_sources = set(train["src"])

# ============================================================
# 8. COVERAGE AUDIT
# ============================================================

test["target_known"] = test["target"].isin(catalogue)

test["source_known"] = test["src"].isin(known_sources)

target_coverage = test["target_known"].mean() if len(test) else 0
source_coverage = test["source_known"].mean() if len(test) else 0

# ============================================================
# 9. VALIDATION SUMMARY
# ============================================================

print("\n========== TRANSITION SUMMARY ==========")

print("Training transitions:", len(train))
print("Validation transitions:", len(val))
print("Testing transitions:", len(test))

print("Training catalogue:", len(catalogue))

print("\nAmbiguous timestamp groups:",
      (groups["event_count"] > 1).sum())

print("Session gap:", SESSION_GAP_MS // 60000, "minutes")

# These are the final two output lines.
print(
    f"\nTEST TARGET COVERAGE: "
    f"{test['target_known'].sum()}/{len(test)} "
    f"({target_coverage:.2%})"
)

print(
    f"TEST SOURCE COVERAGE: "
    f"{test['source_known'].sum()}/{len(test)} "
    f"({source_coverage:.2%})"
)

# ============================================================
# 10. SAVE TRANSITION SPLITS
# ============================================================

OUTPUT_DIR = r"D:\APPS Research\KuaiRand-1K\data"

train.to_csv(
    OUTPUT_DIR + r"\chrono_train_transitions.csv",
    index=False
)

val.to_csv(
    OUTPUT_DIR + r"\chrono_val_transitions.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR + r"\chrono_test_transitions.csv",
    index=False
)

print("\nTransition files saved successfully.")
print("Chronological split analysis complete.")
