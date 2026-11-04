import pandas as pd
from pathlib import Path

folder = Path(r"D:\APPS Research\KuaiRand-1K")

files = [
    "kuairand_video_captions.csv",
    "kuairand_video_categories.csv",
    "video_features_basic_1k.csv"
]

for filename in files:
    path = folder / filename

    print(f"\n{'=' * 55}")
    print("FILE:", filename)

    if not path.exists():
        print("File not found!")
        continue

    df = pd.read_csv(path, nrows=5)

    print("Columns:", df.columns.tolist())
    print("Sample:")
    print(df.head(3).to_string(index=False))