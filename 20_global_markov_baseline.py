
import pandas as pd
from collections import Counter, defaultdict
import math

# ============================================================
# CONFIGURATION
# ============================================================

FILE = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

SESSION_LIMIT = 1800
TRAIN_RATIO = 0.80
K = 10

print("Loading dataset...")

df = pd.read_csv(FILE)

# ============================================================
# 1. SELECT TAB 1 CLICKED EVENTS
# ============================================================

events = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
][["user_id", "video_id", "time_ms"]].copy()

print("Selected events:", len(events))

# ============================================================
# 2. CREATE TIMESTAMP GROUPS
# ============================================================

groups = (
    events.groupby(["user_id", "time_ms"])
    .agg(
        count=("video_id", "size"),
        videos=("video_id", list)
    )
    .reset_index()
    .sort_values(["user_id", "time_ms"])
)

# ============================================================
# 3. PRESERVE AMBIGUOUS GROUPS
# ============================================================

groups["next_count"] = (
    groups.groupby("user_id")["count"].shift(-1)
)

groups["next_time"] = (
    groups.groupby("user_id")["time_ms"].shift(-1)
)

groups["next_videos"] = (
    groups.groupby("user_id")["videos"].shift(-1)
)

groups["gap"] = (
    groups["next_time"] - groups["time_ms"]
) / 1000

# ============================================================
# 4. CREATE CONSERVATIVE TRANSITIONS
# ============================================================

pairs = groups[
    (groups["count"] == 1) &
    (groups["next_count"] == 1) &
    (groups["gap"] >= 0) &
    (groups["gap"] <= SESSION_LIMIT)
].copy()

pairs["video_id"] = pairs["videos"].apply(lambda x: x[0])

pairs["next_video_id"] = (
    pairs["next_videos"].apply(lambda x: x[0])
)

pairs = pairs.sort_values(["user_id", "time_ms"])

print("\nValid transitions:", len(pairs))
print("Users:", pairs["user_id"].nunique())

# ============================================================
# 5. CHRONOLOGICAL 80/20 SPLIT
# ============================================================

train_parts = []
test_parts = []

for user_id, user_data in pairs.groupby("user_id", sort=False):

    user_data = user_data.sort_values("time_ms")

    split_idx = int(len(user_data) * TRAIN_RATIO)

    if split_idx == 0 or split_idx == len(user_data):
        continue

    train_parts.append(user_data.iloc[:split_idx])
    test_parts.append(user_data.iloc[split_idx:])

if not train_parts or not test_parts:
    raise ValueError("Training or testing data is empty.")

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

print("\nTraining transitions:", len(train))
print("Testing transitions:", len(test))
print("Training users:", train["user_id"].nunique())
print("Testing users:", test["user_id"].nunique())

# ============================================================
# 6. ESTIMATE GLOBAL MARKOV TRANSITION COUNTS
# ============================================================

transition_counts = defaultdict(Counter)

for row in train.itertuples(index=False):
    transition_counts[row.video_id][row.next_video_id] += 1

print("\nUnique Markov source states:", len(transition_counts))

# ============================================================
# 7. ESTIMATE POPULARITY FROM TRAINING DATA
# ============================================================

popularity = Counter(train["next_video_id"])

popular_items = [
    video
    for video, count in popularity.most_common(K)
]

catalog = set(popularity.keys())

print("Training candidate catalogue:", len(catalog))

# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def evaluate(recommendations, actual):

    hit = 0
    ndcg = 0
    mrr = 0

    for recs, target in zip(recommendations, actual):

        if target in recs:

            rank = recs.index(target) + 1

            hit += 1
            ndcg += 1 / math.log2(rank + 1)
            mrr += 1 / rank

    total = len(actual)

    if total == 0:
        return 0, 0, 0

    return (
        hit / total,
        ndcg / total,
        mrr / total
    )

# ============================================================
# 9. GENERATE RECOMMENDATIONS
# ============================================================

mc_recommendations = []
pop_recommendations = []

fallback_count = 0
catalog_misses = 0

for row in test.itertuples(index=False):

    source = row.video_id
    target = row.next_video_id

    # Global Markov recommendation
    if source in transition_counts:

        recs = [
            video
            for video, count in
            transition_counts[source].most_common(K)
        ]

    else:

        recs = popular_items.copy()
        fallback_count += 1

    mc_recommendations.append(recs)

    # Popularity baseline
    pop_recommendations.append(popular_items)

    if target not in catalog:
        catalog_misses += 1

actual = test["next_video_id"].tolist()

# ============================================================
# 10. CALCULATE RESULTS
# ============================================================

mc_metrics = evaluate(mc_recommendations, actual)
pop_metrics = evaluate(pop_recommendations, actual)

print("\n========== BASELINE RESULTS ==========")

print("\nGlobal Markov + popularity fallback:")

print(f"Recall@{K}: {mc_metrics[0]:.6f}")
print(f"NDCG@{K}:   {mc_metrics[1]:.6f}")
print(f"MRR@{K}:    {mc_metrics[2]:.6f}")

print("\nPopularity baseline:")

print(f"Recall@{K}: {pop_metrics[0]:.6f}")
print(f"NDCG@{K}:   {pop_metrics[1]:.6f}")
print(f"MRR@{K}:    {pop_metrics[2]:.6f}")

print("\nFallback predictions:", fallback_count)
print("Targets outside training catalog:", catalog_misses)

print("\n=====================================")
