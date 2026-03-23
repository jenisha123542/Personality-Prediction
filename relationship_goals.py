import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\NITRO\Documents\mbti__1.xlsx"

RELATIONSHIP_GOALS = {
    "Long-term relationship": 35,
    "Marriage":               30,
    "Friendship":             20,
    "Casual dating":          15,
}

labels  = list(RELATIONSHIP_GOALS.keys())
weights = np.array(list(RELATIONSHIP_GOALS.values()), dtype=float)
weights /= weights.sum()

df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} rows")

np.random.seed(42)
df['relationship_goal'] = ""

for mbti_type, group in df.groupby('mbti_type'):
    idx = group.index
    df.loc[idx, 'relationship_goal'] = np.random.choice(labels, size=len(idx), p=weights)

print("\nRelationship goal distribution per MBTI type:")
print("-" * 55)
for t in sorted(df['mbti_type'].unique()):
    sub = df[df['mbti_type'] == t]['relationship_goal'].value_counts()
    print(f"  {t:<6} (n={len(df[df['mbti_type']==t]):>3})  " +
          ", ".join(f"{c}:{v}" for c, v in sub.items()))

print(f"\nOverall: {dict(df['relationship_goal'].value_counts())}")

df.to_excel(DATA_PATH, index=False)
print(f"\nSaved -> {DATA_PATH}")