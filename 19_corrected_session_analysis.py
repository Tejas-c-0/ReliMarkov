import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

# Keep every timestamp group, including ambiguous ones
groups = (
    events.groupby(["user_id", "time_ms"])
    .agg(
        event_count=("video_id", "size"),
        video_ids=("video_id", list)
    )
    .reset_index()
    .sort_values(["user_id", "time_ms"])
)

# Identify the immediately following original timestamp group
groups["next_count"] = (
    groups.groupby("user_id")["event_count"].shift(-1)
)

groups["next_time"] = (
    groups.groupby("user_id")["time_ms"].shift(-1)
)

groups["next_videos"] = (
    groups.groupby("user_id")["video_ids"].shift(-1)
)

groups["gap_seconds"] = (
    groups["next_time"] - groups["time_ms"]
) / 1000

# Both immediately adjacent groups must be singletons
pairs = groups[
    (groups["event_count"] == 1) &
    (groups["next_count"] == 1)
].copy()

pairs["video_id"] = pairs["video_ids"].apply(lambda x: x[0])
pairs["next_video_id"] = pairs["next_videos"].apply(lambda x: x[0])

print("\n========== CORRECTED SESSION ANALYSIS ==========")

print("Conservative adjacent pairs:", len(pairs))
print("Users:", pairs["user_id"].nunique())

thresholds = [60, 300, 600, 1800, 3600, 86400]

print("\nThreshold comparison:")

for threshold in thresholds:
    selected = pairs[pairs["gap_seconds"] <= threshold]

    print(
        f"\nThreshold: {threshold} seconds "
        f"({threshold / 60:.0f} minutes)"
    )

    print("Transitions:", len(selected))
    print("Users:", selected["user_id"].nunique())

    print(
        "Retained:",
        round(len(selected) / len(pairs) * 100, 2),
        "%"
    )

print("\nFirst 10 valid pairs:")

print(
    pairs[
        ["user_id", "video_id", "next_video_id", "gap_seconds"]
    ].head(10).to_string(index=False)
)

print("\n================================================")