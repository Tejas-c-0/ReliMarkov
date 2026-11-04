import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")

df = pd.read_csv(file)
df["_row_id"] = range(len(df))

events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

print("\n========== ORDER DIAGNOSTIC ==========")

# Timestamp groups
groups = (
    events.groupby(["user_id", "time_ms"])
    .agg(
        event_count=("video_id", "size"),
        first_row=("_row_id", "min"),
        last_row=("_row_id", "max")
    )
    .reset_index()
)

singletons = groups[groups["event_count"] == 1]
collisions = groups[groups["event_count"] > 1]

print("\nTotal clicked events:", len(events))
print("Unique timestamp groups:", len(groups))
print("Singleton groups:", len(singletons))
print("Collision groups:", len(collisions))

print(
    "Events in singleton groups:",
    singletons["event_count"].sum()
)

print(
    "Percentage of events in collision groups:",
    round(
        collisions["event_count"].sum() / len(events) * 100,
        3
    )
)

# Check original file ordering by user
original_order = events.groupby("user_id")["_row_id"].apply(
    lambda x: x.is_monotonic_increasing
)

print("\nOriginal row order is increasing within users:")
print(original_order.value_counts())

# Inspect a few collision groups in original file order
print("\nExample collision groups in original CSV order:")

sample = collisions.head(5)

for _, group in sample.iterrows():
    rows = events[
        (events["user_id"] == group["user_id"]) &
        (events["time_ms"] == group["time_ms"])
    ].sort_values("_row_id")

    print(
        f"\nUser {group['user_id']} | "
        f"Timestamp {group['time_ms']}"
    )

    print(
        rows[[
            "_row_id",
            "video_id",
            "play_time_ms",
            "duration_ms"
        ]].to_string(index=False)
    )

print("\n=======================================")