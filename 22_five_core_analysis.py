
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

FILE = r"D:\APPS Research\KuaiRand-1K\data\log_standard_4_08_to_4_21_1k.csv"

OUTPUT_FILE = r"D:\APPS Research\KuaiRand-1K\data\kuairand_tab1_5core.csv"

MIN_INTERACTIONS = 5

print("Loading KuaiRand dataset...")

df = pd.read_csv(FILE)

print("\n========== ORIGINAL DATA ==========")
print("Total records:", len(df))

# ==========================================
# 1. SELECT TAB 1 CLICKED INTERACTIONS
# ==========================================

df = df[
    (df["tab"] == 1) &
    (df["is_click"] == 1)
].copy()

print("\n========== SELECTED DATA ==========")
print("Selected interactions:", len(df))
print("Unique users:", df["user_id"].nunique())
print("Unique videos:", df["video_id"].nunique())

# ==========================================
# 2. ITERATIVE 5-CORE FILTERING
# ==========================================

iteration = 0

while True:

    iteration += 1

    item_cnt = df["video_id"].value_counts()
    user_cnt = df["user_id"].value_counts()

    keep = (
        df["video_id"].map(item_cnt).ge(MIN_INTERACTIONS) &
        df["user_id"].map(user_cnt).ge(MIN_INTERACTIONS)
    )

    removed = len(df) - keep.sum()

    print(f"Iteration {iteration}:")
    print("Records removed:", removed)

    if keep.all():
        break

    df = df[keep].copy()

# ==========================================
# 3. FINAL STATISTICS
# ==========================================

print("\n========== 5-CORE RESULTS ==========")

print("Remaining interactions:", len(df))
print("Remaining users:", df["user_id"].nunique())
print("Remaining videos:", df["video_id"].nunique())

print("\n========== REMOVAL STATISTICS ==========")

print("Interactions removed:",
      len(pd.read_csv(FILE, usecols=["user_id", "video_id"])) if False else "See selected-data count above")

# ==========================================
# 4. VALIDATE 5-CORE
# ==========================================

user_counts = df["user_id"].value_counts()
video_counts = df["video_id"].value_counts()

print("\n========== VALIDATION ==========")

print("Minimum user interactions:", user_counts.min())
print("Minimum video interactions:", video_counts.min())

assert user_counts.min() >= MIN_INTERACTIONS
assert video_counts.min() >= MIN_INTERACTIONS

print("5-core validation: PASSED")

# ==========================================
# 5. SAVE FILTERED DATA
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nFiltered dataset saved successfully:")
print(OUTPUT_FILE)

print("\n========== ANALYSIS COMPLETE ==========")
