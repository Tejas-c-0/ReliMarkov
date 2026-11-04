
import pandas as pd
from collections import Counter, defaultdict
import math

# ============================================================
# CONFIGURATION
# ============================================================

FILE = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

K = 10
SESSION_LIMIT = 1800
TRAIN_RATIO = 0.80

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
# 3. PRESERVE AMBIGUOUS TIMESTAMP GROUPS
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
# 5. CHRONOLOGICAL 80/20 USER-WISE SPLIT
# ============================================================

train_parts = []
test_parts = []

for user_id, user_data in pairs.groupby("user_id", sort=False):

    user_data = user_data.sort_values("time_ms")

    split_idx = int(len(user_data) * TRAIN_RATIO)

    # Same splitting rule as Script 20
    if split_idx == 0 or split_idx == len(user_data):
        continue

    train_parts.append(user_data.iloc[:split_idx])
    test_parts.append(user_data.iloc[split_idx:])

if not train_parts or not test_parts:
    raise ValueError("Training or testing data is empty.")

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

print("\n========== DATA SPLIT ==========")
print("Training transitions:", len(train))
print("Testing transitions:", len(test))
print("Training users:", train["user_id"].nunique())
print("Testing users:", test["user_id"].nunique())

# ============================================================
# 6. BUILD TRAINING-ONLY MARKOV MODEL
# ============================================================

transition_counts = defaultdict(Counter)

for row in train.itertuples(index=False):
    transition_counts[row.video_id][row.next_video_id] += 1

print("\nUnique Markov source states:", len(transition_counts))

# ============================================================
# 7. TRAINING-ONLY POPULARITY BASELINE
# ============================================================

popularity = Counter(train["next_video_id"])

popular_items = [
    video
    for video, count in popularity.most_common(K)
]

catalog = set(popularity.keys())

print("Training candidate catalogue:", len(catalog))

# ============================================================
# 8. IDENTIFY TEST COVERAGE
# ============================================================

test["target_known"] = test["next_video_id"].isin(catalog)

test["source_known"] = test["video_id"].isin(transition_counts)

both_known = (
    test["target_known"] &
    test["source_known"]
)

print("\n========== COVERAGE AUDIT ==========")

print("\nTest coverage:")

print("Target known:", test["target_known"].sum())
print("Target unknown:", (~test["target_known"]).sum())

print("Source known:", test["source_known"].sum())
print("Source unknown:", (~test["source_known"]).sum())

print("Both source and target known:", both_known.sum())

# ============================================================
# 9. GENERATE RECOMMENDATIONS
# ============================================================

markov_recs = []
pop_recs = []

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

    markov_recs.append(recs)

    # Popularity recommendation
    pop_recs.append(popular_items)

    if target not in catalog:
        catalog_misses += 1

print("\nFallback predictions:", fallback_count)
print("Targets outside training catalogue:", catalog_misses)

# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate(indices, recommendations):

    hits = 0
    ndcg = 0
    mrr = 0

    rank_counts = Counter()

    for idx in indices:

        target = test.iloc[idx]["next_video_id"]
        recs = recommendations[idx]

        if target in recs:

            rank = recs.index(target) + 1

            hits += 1
            ndcg += 1 / math.log2(rank + 1)
            mrr += 1 / rank

            rank_counts[rank] += 1

    total = len(indices)

    if total == 0:
        return None

    return {
        "cases": total,
        "hits": hits,
        "recall": hits / total,
        "ndcg": ndcg / total,
        "mrr": mrr / total,
        "hit_ranks": dict(sorted(rank_counts.items()))
    }

# ============================================================
# 11. EVALUATION SUBSETS
# ============================================================

test = test.reset_index(drop=True)

all_idx = list(range(len(test)))

target_idx = test.index[
    test["target_known"]
].tolist()

supported_idx = test.index[
    test["target_known"] &
    test["source_known"]
].tolist()

# ============================================================
# 12. PRINT FINAL RESULTS
# ============================================================

for name, indices in [
    ("ALL TEST CASES", all_idx),
    ("TARGET IN TRAINING CATALOGUE", target_idx),
    ("SOURCE AND TARGET KNOWN", supported_idx)
]:

    print(f"\n--- {name} ---")

    for model_name, recommendations in [
        ("Markov + fallback", markov_recs),
        ("Popularity", pop_recs)
    ]:

        result = evaluate(indices, recommendations)

        print(f"\n{model_name}:")

        if result is None:
            print("No eligible cases")
            continue

        print("Cases:", result["cases"])
        print("Hits:", result["hits"])

        print(f"Recall@{K}: {result['recall']:.6f}")
        print(f"NDCG@{K}:   {result['ndcg']:.6f}")
        print(f"MRR@{K}:    {result['mrr']:.6f}")

        print("Hit ranks:", result["hit_ranks"])

print("\n========== AUDIT COMPLETE ==========")
