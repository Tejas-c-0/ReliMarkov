import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

# Select Tab 1 clicked events
events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

events = events.sort_values(["user_id", "time_ms"])

print("\n========== TAB 1 SEQUENCE QUALITY ==========")

print("Total events:", len(events))
print("Unique users:", events["user_id"].nunique())
print("Unique videos:", events["video_id"].nunique())

# Calculate time gaps within each user's sequence
events["gap_seconds"] = (
    events.groupby("user_id")["time_ms"]
    .diff() / 1000
)

gaps = events["gap_seconds"].dropna()

print("\nTime gap statistics (seconds):")
print(
    gaps.quantile(
        [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]
    )
)

for threshold in [60, 300, 1800, 3600, 86400]:
    count = (gaps > threshold).sum()
    print(f"Gaps > {threshold} seconds: {count:,}")

# Repeated consecutive videos
events["previous_video"] = (
    events.groupby("user_id")["video_id"].shift(1)
)

repeats = (
    events["video_id"] == events["previous_video"]
).sum()

print("\nConsecutive repeated videos:", repeats)

# Check timestamp ordering
ordering = events.groupby("user_id")["time_ms"].apply(
    lambda x: x.is_monotonic_increasing
)

print("\nUsers with chronological ordering:", ordering.sum())
print("Users with invalid ordering:", (~ordering).sum())

# Inspect user history distribution
history = events.groupby("user_id").size()

print("\nEvents per user:")
print(history.describe(percentiles=[
    0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99
]))

print("\n==========================================")