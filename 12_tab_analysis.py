import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

print("\n========== TAB BEHAVIOUR ANALYSIS ==========")

# Interaction distribution by tab
tab_stats = df.groupby("tab").agg(
    records=("user_id", "size"),
    users=("user_id", "nunique"),
    videos=("video_id", "nunique"),
    click_rate=("is_click", "mean"),
    long_view_rate=("long_view", "mean"),
    like_rate=("is_like", "mean"),
    forward_rate=("is_forward", "mean"),
    hate_rate=("is_hate", "mean"),
    random_rate=("is_rand", "mean")
)

print("\nTab statistics:")
print(tab_stats.round(4).to_string())

# Interaction combinations by tab
print("\nClick and long-view combinations by tab:")

for tab in sorted(df["tab"].unique()):
    subset = df[df["tab"] == tab]

    print(f"\nTab {tab} | Records: {len(subset):,}")

    print(pd.crosstab(
        subset["is_click"],
        subset["long_view"]
    ))

# Watch-time ratio by tab
valid = df[df["duration_ms"] > 0].copy()
valid["watch_ratio"] = (
    valid["play_time_ms"] / valid["duration_ms"]
)

print("\nWatch ratio by tab:")
print(
    valid.groupby("tab")["watch_ratio"]
    .agg(["count", "median", "mean"])
    .round(3)
    .to_string()
)

print("\n============================================")