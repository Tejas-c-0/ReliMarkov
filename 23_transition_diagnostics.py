
import pandas as pd
import numpy as np

FILE = r"D:\APPS Research\KuaiRand-1K\data\kuairand_tab1_5core.csv"

print("Loading dataset...")
df = pd.read_csv(FILE)

# ==========================================
# 1. CHRONOLOGICAL EVENT ORDER
# ==========================================

# Stable sorting preserves original CSV order for tied timestamps.
# It does NOT establish the true order of simultaneous events.
d = df.sort_values(
    ["user_id", "time_ms"],
    kind="mergesort"
).copy()

# ==========================================
# 2. TIME GAP DISTRIBUTION
# ==========================================

gap = (
    d.groupby("user_id")["time_ms"]
    .diff()
    .dropna() / 60000
)

print("\n========== TIME GAP DISTRIBUTION ==========")

print(
    gap.describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n========== SESSION GAP COVERAGE ==========")

for minutes in [30, 120, 360, 1440]:
    print(
        f"Gap <= {minutes} minutes:",
        round((gap <= minutes).mean(), 4)
    )

print(
    "Zero-gap share:",
    round((gap == 0).mean(), 4)
)

# ==========================================
# 3. TIMESTAMP COLLISIONS
# ==========================================

timestamp_groups = (
    d.groupby(["user_id", "time_ms"])
    .size()
)

print("\n========== TIMESTAMP COLLISIONS ==========")

print("Total timestamp groups:", len(timestamp_groups))

print(
    "Groups containing multiple events:",
    (timestamp_groups > 1).sum()
)

print(
    "Events in ambiguous groups:",
    timestamp_groups[timestamp_groups > 1].sum()
)

# ==========================================
# 4. REBUILD THE CHRONOLOGICAL CUTOFFS
# ==========================================

t1, t2 = d["time_ms"].quantile([0.70, 0.85])

training_events = d[d["time_ms"] <= t1]

training_catalogue = set(training_events["video_id"])

testing_events = d[d["time_ms"] > t2]

print("\n========== VIDEO CATALOGUE ==========")

print(
    "Videos in training events:",
    len(training_catalogue)
)

print(
    "Unique videos in test events:",
    testing_events["video_id"].nunique()
)

unseen_test = ~testing_events["video_id"].isin(
    training_catalogue
)

print(
    "Test event videos unseen in training:",
    round(unseen_test.mean(), 4)
)

# ==========================================
# 5. LOAD EXISTING TRANSITION FILES
# ==========================================

TRAIN_FILE = (
    r"D:\APPS Research\KuaiRand-1K\data"
    r"\chrono_train_transitions.csv"
)

TEST_FILE = (
    r"D:\APPS Research\KuaiRand-1K\data"
    r"\chrono_test_transitions.csv"
)

tr_train = pd.read_csv(TRAIN_FILE)
tr_test = pd.read_csv(TEST_FILE)

print("\n========== TRANSITION SOURCE FREQUENCY ==========")

source_frequency = tr_train.groupby("src").size()

print(source_frequency.describe())

print(
    "Sources appearing once:",
    (source_frequency == 1).sum()
)

print(
    "Sources appearing more than 5 times:",
    (source_frequency > 5).sum()
)

print("\n========== TRANSITION TARGET COVERAGE ==========")

print(
    "Test targets unseen in training event catalogue:",
    round(
        (~tr_test["target"].isin(training_catalogue)).mean(),
        4
    )
)

print("\n========== DIAGNOSTICS COMPLETE ==========")
