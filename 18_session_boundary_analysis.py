import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

# Preserve timestamp groups
groups = (
    events.groupby(["user_id", "time_ms"])
    .agg(
        event_count=("video_id", "size"),
        video_ids=("video_id", list)
    )
    .reset_index()
    .sort_values(["user_id", "time_ms"])
)

# Keep only unique-event timestamp groups
groups = groups[groups["event_count"] == 1].copy()

groups["next_time"] = (
    groups.groupby("user_id")["time_ms"].shift(-1)
)

groups["next_video"] = (
    groups.groupby("user_id")["video_ids"].shift(-1)
)

groups["gap_seconds"] = (
    groups["next_time"] - groups["time_ms"]
) / 1000

# Keep adjacent singleton pairs
pairs = groups.dropna(
    subset=["next_time", "next_video"]
).copy()

pairs["video_id"] = pairs["video_ids"].apply(lambda x: x[0])
pairs["next_video_id"] = pairs["next_video"].apply(lambda x: x[0])

print("\n========== SESSION BOUNDARY ANALYSIS ==========")

print("Total conservative pairs:", len(pairs))

thresholds = [
    60,
    300,
    600,
    1800,
    3600,
    86400
]

print("\nThreshold comparison:")

for threshold in thresholds:
    filtered = pairs[
        pairs["gap_seconds"] <= threshold
    ]

    print(
        f"\nThreshold: {threshold} seconds "
        f"({threshold / 60:.0f} minutes)"
    )

    print("Transitions:", len(filtered))
    print("Users:", filtered["user_id"].nunique())
    print(
        "Percentage retained:",
        round(len(filtered) / len(pairs) * 100, 2),
        "%"
    )

print("\n===============================================")