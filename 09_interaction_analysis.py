import pandas as pd

file = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

columns = [
    "is_click",
    "is_like",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms"
]

print("Loading dataset...")

df = pd.read_csv(file)

print("\n========== INTERACTION ANALYSIS ==========")

print("Dataset shape:", df.shape)

print("\nInteraction statistics:")
print(df[columns].describe().to_string())

for col in columns[:5]:
    print(f"\n{col} distribution:")
    print(df[col].value_counts(dropna=False))

print("\nWatch-time ratio:")

valid = df[df["duration_ms"] > 0].copy()
valid["watch_ratio"] = (
    valid["play_time_ms"] / valid["duration_ms"]
)

print(valid["watch_ratio"].describe())

print("\n==========================================")