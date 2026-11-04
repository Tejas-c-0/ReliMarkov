import pandas as pd
from pathlib import Path

DATA = Path("data")

files = [
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv"
]

frames = []

for filename in files:
    df = pd.read_csv(DATA / filename, nrows=10000)
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# Count clicks in each user-timestamp batch
batch_clicks = df.groupby(
    ["user_id", "time_ms"]
)["is_click"].sum().reset_index(name="click_count")

# Retain batches with exactly one click
valid_batches = batch_clicks[
    batch_clicks["click_count"] == 1
][["user_id", "time_ms"]]

# Extract the clicked video from each valid batch
events = df[
    (df["is_click"] == 1)
].merge(
    valid_batches,
    on=["user_id", "time_ms"]
)

# Sort chronologically
events = events.sort_values(
    ["user_id", "time_ms"]
).reset_index(drop=True)

# Construct consecutive transitions
events["next_video_id"] = (
    events.groupby("user_id")["video_id"].shift(-1)
)

transitions = events.dropna(
    subset=["next_video_id"]
).copy()

print("Valid clicked events:", len(events))
print("Users:", events["user_id"].nunique())
print("Transitions:", len(transitions))

print("\nFirst 20 events:")
print(events[[
    "user_id", "video_id", "time_ms",
    "play_time_ms", "duration_ms"
]].head(20).to_string(index=False))

print("\nFirst 20 transitions:")
print(transitions[[
    "user_id", "video_id", "next_video_id"
]].head(20).to_string(index=False))

# Save pilot outputs
events.to_csv("processed_events.csv", index=False)
transitions.to_csv("pilot_transitions.csv", index=False)

print("\nSaved processed_events.csv")
print("Saved pilot_transitions.csv")