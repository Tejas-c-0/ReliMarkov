import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

events = events.sort_values(["user_id", "time_ms"])

print("\n========== TIMESTAMP COLLISION ANALYSIS ==========")

# Number of events sharing the same timestamp per user
timestamp_groups = (
    events.groupby(["user_id", "time_ms"])
    .size()
    .reset_index(name="events_at_timestamp")
)

print("\nTimestamp group statistics:")
print(timestamp_groups["events_at_timestamp"].describe())

# Count timestamps containing multiple events
collisions = timestamp_groups[
    timestamp_groups["events_at_timestamp"] > 1
]

print("\nTimestamp groups with multiple events:", len(collisions))
print("Events in collision groups:", collisions["events_at_timestamp"].sum())

print("\nLargest timestamp groups:")
print(
    collisions.sort_values(
        "events_at_timestamp",
        ascending=False
    ).head(15).to_string(index=False)
)

# Inspect sample collisions
print("\nSample collision events:")

sample_keys = collisions.head(5)[["user_id", "time_ms"]]

for _, row in sample_keys.iterrows():
    sample = events[
        (events["user_id"] == row["user_id"]) &
        (events["time_ms"] == row["time_ms"])
    ]

    print("\nUser:", row["user_id"], "| Timestamp:", row["time_ms"])
    print(
        sample[[
            "video_id",
            "play_time_ms",
            "duration_ms",
            "is_like",
            "is_forward"
        ]].to_string(index=False)
    )

# Count distinct timestamps
print("\nTotal clicked events:", len(events))
print("Unique user-timestamp groups:", len(timestamp_groups))

print("\n===================================================")