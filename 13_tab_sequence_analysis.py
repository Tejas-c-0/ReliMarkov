import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

print("\n========== TAB SEQUENCE ANALYSIS ==========")

# Extract clicked events
clicked = df[df["is_click"] == 1].copy()

# Sort chronologically
clicked = clicked.sort_values(["user_id", "time_ms"])

# Analyse each tab
stats = clicked.groupby("tab").agg(
    clicked_events=("video_id", "size"),
    unique_users=("user_id", "nunique"),
    unique_videos=("video_id", "nunique")
)

# Number of clicked events per user and tab
user_counts = (
    clicked.groupby(["tab", "user_id"])
    .size()
    .reset_index(name="events")
)

sequence_stats = user_counts.groupby("tab").agg(
    users=("user_id", "nunique"),
    median_events_per_user=("events", "median"),
    mean_events_per_user=("events", "mean"),
    max_events_per_user=("events", "max")
)

result = stats.join(sequence_stats, how="left")

print("\nClicked sequence statistics:")
print(result.round(2).to_string())

# Potential transitions per tab
result["potential_transitions"] = (
    result["clicked_events"] - result["unique_users"]
)

print("\nPotential transitions by tab:")
print(
    result[[
        "clicked_events",
        "unique_users",
        "potential_transitions"
    ]].to_string()
)

print("\n============================================")