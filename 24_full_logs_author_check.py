import pandas as pd, numpy as np

cols = ["user_id","video_id","time_ms","tab","is_click","is_like","is_hate",
        "is_forward","long_view","play_time_ms","duration_ms"]
files = ["log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv"]
df = pd.concat([pd.read_csv(rf"data\{f}", usecols=cols) for f in files], ignore_index=True)
df["row"] = df.index
print("raw events:", len(df))

df = df[(df.tab == 1) & (df.is_click == 1)]
print("tab==1 & is_click==1:", len(df))

wr = (df.play_time_ms / df.duration_ms.replace(0, np.nan)).clip(0, 1)
print(wr.describe(percentiles=[.05, .25, .5, .75, .95]))
print("long_view:", round(df.long_view.mean(), 3), "| like:", round(df.is_like.mean(), 4),
      "| forward:", round(df.is_forward.mean(), 4), "| hate:", round(df.is_hate.mean(), 4))

vf = (pd.read_csv(r"data\video_features_basic_1k.csv", usecols=["video_id", "author_id"])
        .drop_duplicates("video_id"))
df = df.merge(vf, on="video_id", how="left")
print("events with author_id:", round(df.author_id.notna().mean(), 3))

df = df.sort_values(["user_id", "time_ms", "row"], kind="mergesort")
t1, t2 = df.time_ms.quantile([0.70, 0.85])
print("train ends:", pd.to_datetime(t1, unit="ms"), "| test starts:", pd.to_datetime(t2, unit="ms"))
train, test = df[df.time_ms <= t1], df[df.time_ms > t2]

def trans(d, col, gap_ms=30*60*1000):
    g = d.groupby("user_id")
    out = pd.DataFrame({"src": g[col].shift(1), "tgt": d[col],
                        "gap": d["time_ms"] - g["time_ms"].shift(1)})
    return out[out.src.notna() & out.tgt.notna() & (out.gap <= gap_ms)]

for col in ["video_id", "author_id"]:
    tr, te = trans(train, col), trans(test, col)
    print(f"\n{col}: train events {len(train)} | train transitions {len(tr)} | test transitions {len(te)}")
    print("  zero-gap share of train transitions:", round((tr.gap == 0).mean(), 3))
    print("  unique states in train:", train[col].nunique())
    print("  test targets seen in train:", round(te.tgt.isin(set(train[col])).mean(), 3))
    print("  test sources seen in train:", round(te.src.isin(set(tr.src)).mean(), 3))
    print("  source repeat counts:", tr.groupby("src").size().describe()[["50%", "75%", "max"]].to_dict())