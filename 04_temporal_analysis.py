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

# Sort by actual timestamps
df = df.sort_values(
    ["user_id", "time_ms"]
).copy()

# Group identical timestamps into batches
batches = df.groupby(
    ["user_id", "time_ms"],
    as_index=False
).agg(
    records=("video_id", "size"),
    clicks=("is_click", "sum"),
    play_count=("play_time_ms", lambda x: (x > 0).sum())
)

# Calculate time between batches for each user
batches = batches.sort_values(
    ["user_id", "time_ms"]
)

batches["gap_seconds"] = (
    batches.groupby("user_id")["time_ms"].diff() / 1000
)

print("\n--- Time gap statistics (seconds) ---")
print(batches["gap_seconds"].describe())

print("\n--- First 30 batches ---")
print(
    batches[[
        "user_id", "time_ms", "records",
        "clicks", "play_count", "gap_seconds"
    ]].head(30).to_string(index=False)
)

print("\n--- Time gap distribution ---")
print(
    pd.cut(
        batches["gap_seconds"].dropna(),
        bins=[0, 1, 5, 30, 60, 300, 3600, float("inf")]
    ).value_counts().sort_index()
)