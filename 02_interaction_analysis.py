import pandas as pd
from pathlib import Path

DATA = Path("data")

# Load a small sample from both standard logs
df1 = pd.read_csv(
    DATA / "log_standard_4_08_to_4_21_1k.csv",
    nrows=10000
)

df2 = pd.read_csv(
    DATA / "log_standard_4_22_to_5_08_1k.csv",
    nrows=10000
)

df = pd.concat([df1, df2], ignore_index=True)

# Sort chronologically
df = df.sort_values(
    ["user_id", "time_ms", "video_id"]
).reset_index(drop=True)

print("Total sampled records:", len(df))
print("Unique users:", df["user_id"].nunique())
print("Unique videos:", df["video_id"].nunique())

print("\n--- Feedback summary ---")

signals = [
    "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_hate",
    "long_view"
]

for col in signals:
    print(f"{col}: {df[col].sum()}")

print("\n--- Play time statistics ---")
print(df["play_time_ms"].describe())

print("\n--- Duration statistics ---")
print(df["duration_ms"].describe())

# Check simultaneous records for each user
same_time = df.groupby(
    ["user_id", "time_ms"]
).size()

print("\n--- Timestamp collision analysis ---")
print("User-timestamp groups:", len(same_time))
print("Groups containing multiple records:", (same_time > 1).sum())
print("Maximum records at one user timestamp:", same_time.max())

print("\n--- Sample interactions ---")
print(
    df[[
        "user_id", "video_id", "time_ms",
        "is_click", "play_time_ms", "duration_ms"
    ]].head(20).to_string(index=False)
)