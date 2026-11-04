import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

# Aggregate events sharing a user and timestamp
groups = (
    events.groupby(["user_id", "time_ms"])
    .agg(
        event_count=("video_id", "size"),
        video_ids=("video_id", list)
    )
    .reset_index()
)

groups = groups.sort_values(["user_id", "time_ms"])

# A group is unambiguous only if it contains one clicked video
groups["is_unambiguous"] = groups["event_count"] == 1

# Examine consecutive timestamp groups within each user
groups["next_event_count"] = (
    groups.groupby("user_id")["event_count"].shift(-1)
)

groups["next_is_unambiguous"] = (
    groups.groupby("user_id")["is_unambiguous"].shift(-1)
)

groups["gap_seconds"] = (
    groups.groupby("user_id")["time_ms"].shift(-1)
    - groups["time_ms"]
) / 1000

# Count only adjacent groups where both contain exactly one event
valid_pairs = groups[
    groups["is_unambiguous"] &
    groups["next_is_unambiguous"]
].copy()

print("\n========== CONSERVATIVE TRANSITION ANALYSIS ==========")

print("Total timestamp groups:", len(groups))
print("Unambiguous groups:", groups["is_unambiguous"].sum())
print("Ambiguous groups:", (~groups["is_unambiguous"]).sum())

print("\nAdjacent unambiguous pairs:", len(valid_pairs))

print("\nPotential pairs by user:")
print(
    valid_pairs.groupby("user_id").size().describe()
)

print("\nGap statistics for valid pairs:")
print(valid_pairs["gap_seconds"].describe())

# Candidate transitions, without crossing ambiguous groups
valid_pairs["video_id"] = valid_pairs["video_ids"].apply(lambda x: x[0])

next_videos = (
    groups.groupby("user_id")["video_ids"].shift(-1)
)

valid_pairs["next_video_id"] = (
    next_videos.loc[valid_pairs.index].apply(lambda x: x[0])
)

print("\nFirst 10 conservative transitions:")
print(
    valid_pairs[
        ["user_id", "time_ms", "video_id", "next_video_id", "gap_seconds"]
    ].head(10).to_string(index=False)
)

print("\n======================================================")