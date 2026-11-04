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

# Treat same-user, same-timestamp records as a batch
batches = df.groupby(["user_id", "time_ms"]).agg(
    records=("video_id", "size"),
    clicked=("is_click", "sum"),
    positive_play=("play_time_ms", lambda x: (x > 0).sum()),
    liked=("is_like", "sum"),
    forwarded=("is_forward", "sum")
).reset_index()

print("\nTotal timestamp batches:", len(batches))

print("\nRecords per batch:")
print(batches["records"].value_counts().sort_index())

print("\nNumber of clicked videos per batch:")
print(batches["clicked"].value_counts().sort_index())

print("\nNumber of positive-play videos per batch:")
print(batches["positive_play"].value_counts().sort_index())

print("\nBatches with exactly one click:")
print((batches["clicked"] == 1).sum())

print("\nBatches with multiple clicks:")
print((batches["clicked"] > 1).sum())

print("\nExample batches:")
print(batches.head(20).to_string(index=False))