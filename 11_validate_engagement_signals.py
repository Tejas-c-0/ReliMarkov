import pandas as pd
import numpy as np

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

print("Loading dataset...")
df = pd.read_csv(file)

print("\n========== SIGNAL VALIDATION ==========")

# Exclude invalid durations for duration-based checks
valid = df[df["duration_ms"] > 0].copy()

# Reconstruct the documented long_view rule
valid["expected_long_view"] = np.where(
    valid["duration_ms"] <= 18000,
    valid["play_time_ms"] >= valid["duration_ms"],
    valid["play_time_ms"] >= 18000
).astype(int)

# Compare reconstructed and recorded values
valid["long_view_match"] = (
    valid["expected_long_view"] == valid["long_view"]
)

print("\nLong-view rule verification:")
print("Valid records:", len(valid))
print("Matching records:", valid["long_view_match"].sum())
print("Mismatching records:", (~valid["long_view_match"]).sum())
print(
    "Match percentage:",
    round(valid["long_view_match"].mean() * 100, 4)
)

# Compare click and long_view by tab
print("\nClick and long-view rates by tab:")

tab_stats = (
    df.groupby("tab")
    .agg(
        records=("is_click", "size"),
        click_rate=("is_click", "mean"),
        long_view_rate=("long_view", "mean")
    )
)

print(tab_stats.round(4).to_string())

# Check interaction signal overlap
print("\nSignal combinations:")

print(
    pd.crosstab(
        df["is_click"],
        df["long_view"],
        margins=True
    )
)

# Compare click rates across duration groups
valid["duration_group"] = pd.cut(
    valid["duration_ms"],
    bins=[0, 7000, 18000, float("inf")],
    labels=["<=7 sec", "7-18 sec", ">18 sec"]
)

print("\nClick and long-view rates by video duration:")

duration_stats = (
    valid.groupby("duration_group", observed=False)
    .agg(
        records=("is_click", "size"),
        click_rate=("is_click", "mean"),
        long_view_rate=("long_view", "mean")
    )
)

print(duration_stats.round(4).to_string())

print("\n=======================================")