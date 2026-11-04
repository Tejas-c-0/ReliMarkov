import pandas as pd
import numpy as np

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

df = pd.read_csv(file)

print("\n========== WATCH-TIME INVESTIGATION ==========")

# 1. Duration validity
print("\nDuration <= 0:", (df["duration_ms"] <= 0).sum())

# 2. Calculate ratio only for valid durations
valid = df[df["duration_ms"] > 0].copy()

valid["watch_ratio"] = (
    valid["play_time_ms"] / valid["duration_ms"]
)

# 3. Examine ratio distribution
print("\nWatch ratio quantiles:")
print(
    valid["watch_ratio"].quantile(
        [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1]
    )
)

# 4. Count extreme ratios
for threshold in [1, 2, 5, 10]:
    count = (valid["watch_ratio"] > threshold).sum()
    percentage = count / len(valid) * 100

    print(
        f"Ratio > {threshold}: {count:,} "
        f"({percentage:.3f}%)"
    )

# 5. Compare long_view with watch ratio
print("\nWatch ratio by long_view:")
print(
    valid.groupby("long_view")["watch_ratio"]
    .agg(["count", "mean", "median"])
)

# 6. Compare click behaviour
print("\nClick vs long_view:")
print(
    pd.crosstab(
        df["is_click"],
        df["long_view"],
        normalize="index"
    ).round(4)
)

# 7. Check whether engagement signals overlap
print("\nEngagement signal correlations:")
signals = [
    "is_click",
    "is_like",
    "is_forward",
    "is_hate",
    "long_view"
]

print(df[signals].corr().round(3))

print("\n==============================================")