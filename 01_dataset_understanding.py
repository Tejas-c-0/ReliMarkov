import pandas as pd
from pathlib import Path

DATA = Path("data")

files = [
    "log_random_4_22_to_5_08_1k.csv",
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv"
]

for filename in files:
    df = pd.read_csv(DATA / filename, nrows=10000)

    print("\n", "=" * 60)
    print(filename)
    print("=" * 60)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample records:")
    print(df.head(5).to_string(index=False))