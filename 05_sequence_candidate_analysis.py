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

# Analyse each user-timestamp batch
batch = df.groupby(
    ["user_id", "time_ms"]
).agg(
    records=("video_id", "size"),
    clicked=("is_click", "sum"),
    positive_play=("play_time_ms", lambda x: (x > 0).sum()),
    liked=("is_like", "sum"),
    forwarded=("is_forward", "sum"),
    hated=("is_hate", "sum")
).reset_index()

# Classify batches by click count
batch["click_status"] = batch["clicked"].apply(
    lambda x: "No click" if x == 0
    else "One click" if x == 1
    else "Multiple clicks"
)

print("\n--- Click classification ---")
print(batch["click_status"].value_counts())

print("\n--- Clicks vs positive play ---")
print(pd.crosstab(
    batch["click_status"],
    batch["positive_play"] > 0
))

# Inspect batches with exactly one click
single_click_times = batch.loc[
    batch["clicked"] == 1,
    ["user_id", "time_ms"]
]

single_click_records = df.merge(
    single_click_times,
    on=["user_id", "time_ms"]
)

print("\n--- Examples of single-click batches ---")

for (user, timestamp), group in single_click_records.groupby(
    ["user_id", "time_ms"]
):
    print("\nUser:", user, "| Timestamp:", timestamp)

    print(group[[
        "video_id",
        "is_click",
        "play_time_ms",
        "duration_ms",
        "is_like",
        "is_forward"
    ]].to_string(index=False))

    if timestamp == single_click_records["time_ms"].iloc[-1]:
        break