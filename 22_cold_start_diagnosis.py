
import pandas as pd
from collections import Counter

FILE_PATH = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"
SESSION_GAP_SECONDS = 1800
TRAIN_RATIO = 0.80

print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

# Reconstruct the same interaction subset used in Script 20.
df = df[(df["tab"] == 1) & (df["is_click"] == 1)].copy()
df = df.sort_values(["user_id", "time_ms"])

# Preserve every timestamp group, including ambiguous groups.
groups_by_user = []

for user_id, user_df in df.groupby("user_id", sort=False):
    timestamp_groups = (
        user_df.groupby("time_ms", sort=True)
        .agg(
            event_count=("video_id", "size"),
            video_id=("video_id", "first")
        )
        .reset_index()
    )

    groups = list(timestamp_groups.itertuples(index=False))
    user_transitions = []

    # Only adjacent singleton timestamp groups are eligible.
    for i in range(len(groups) - 1):
        current = groups[i]
        following = groups[i + 1]

        if current.event_count != 1 or following.event_count != 1:
            continue

        gap_seconds = (following.time_ms - current.time_ms) / 1000

        if 0 <= gap_seconds <= SESSION_GAP_SECONDS:
            user_transitions.append({
                "user_id": user_id,
                "video_id": current.video_id,
                "next_video_id": following.video_id
            })

    if user_transitions:
        groups_by_user.append(pd.DataFrame(user_transitions))

transitions = pd.concat(groups_by_user, ignore_index=True)

# Same chronological per-user 80/20 split.
train_parts = []
test_parts = []

for user_id, user_df in transitions.groupby("user_id", sort=False):
    split_idx = int(len(user_df) * TRAIN_RATIO)

    if split_idx == 0 or split_idx == len(user_df):
        continue
    if split_idx == 0 or split_idx == len(user_df):
        continue

    train_parts.append(user_df.iloc[:split_idx])
    test_parts.append(user_df.iloc[split_idx:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

# Item coverage.
train_sources = set(train["video_id"])
train_targets = set(train["next_video_id"])
train_items = train_sources | train_targets

test_sources = set(test["video_id"])
test_targets = set(test["next_video_id"])

print("\n========== ITEM COVERAGE ==========")
print("Training transitions:", len(train))
print("Testing transitions:", len(test))

print("Unique training source videos:", len(train_sources))
print("Unique training target videos:", len(train_targets))
print("Unique training videos (union):", len(train_items))

print("Unique testing source videos:", len(test_sources))
print("Unique testing target videos:", len(test_targets))

print("Test targets seen in training:", len(test_targets & train_items))
print("Test targets unseen in training:", len(test_targets - train_items))

print("Test sources seen in training:", len(test_sources & train_items))
print("Test sources unseen in training:", len(test_sources - train_items))

# Transition frequency distribution.
source_counts = Counter(train["video_id"])
target_counts = Counter(train["next_video_id"])

print("\n========== TRANSITION FREQUENCY ==========")

for name, counts in [
    ("Source videos", source_counts),
    ("Target videos", target_counts)
]:
    frequencies = list(counts.values())
    frequencies.sort()

    print(f"\n{name}")
    print("Unique videos:", len(frequencies))
    print("Videos with only 1 occurrence:",
          sum(x == 1 for x in frequencies))
    print("Videos with 2-5 occurrences:",
          sum(2 <= x <= 5 for x in frequencies))
    print("Videos with more than 5 occurrences:",
          sum(x > 5 for x in frequencies))

    print("Median occurrences:", pd.Series(frequencies).median())
    print("Maximum occurrences:", max(frequencies))

# Per-user training history.
user_counts = train.groupby("user_id").size()

print("\n========== USER HISTORY ==========")
print("Training users:", len(user_counts))
print("Median transitions per user:", user_counts.median())
print("Mean transitions per user:", user_counts.mean())
print("Users with fewer than 10 transitions:",
      (user_counts < 10).sum())
print("Users with fewer than 30 transitions:",
      (user_counts < 30).sum())

print("\n========== DIAGNOSIS COMPLETE ==========")