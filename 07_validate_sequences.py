import pandas as pd

events = pd.read_csv("processed_events.csv")
transitions = pd.read_csv("pilot_transitions.csv")

print("\n========== DATASET VALIDATION ==========")

print("Total events:", len(events))
print("Unique users:", events["user_id"].nunique())
print("Unique videos:", events["video_id"].nunique())

print("\nEvents per user:")
print(events.groupby("user_id").size())

print("\nMissing values:")
print(events.isnull().sum())

print("\nZero or invalid durations:")
print((events["duration_ms"] <= 0).sum())

print("\nDuplicate user-video-time events:")
print(
    events.duplicated(
        subset=["user_id", "video_id", "time_ms"]
    ).sum()
)

print("\nTimestamp ordering:")
print(
    events.groupby("user_id")["time_ms"]
    .apply(lambda x: x.is_monotonic_increasing)
)

print("\nTransition validation:")
print("Total transitions:", len(transitions))
print(
    "Transitions with missing next video:",
    transitions["next_video_id"].isna().sum()
)

print("\nFirst 10 transitions:")
print(transitions.head(10))

print("\n========================================")