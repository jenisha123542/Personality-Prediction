import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\NITRO\Documents\mbti__1.xlsx"

NEPAL_CITIES = {
    "Kathmandu":   15,
    "Pokhara":     10,
    "Lalitpur":     8,
    "Biratnagar":   6,
    "Bharatpur":    5,
    "Birgunj":      5,
    "Dharan":       4,
    "Hetauda":      3,
    "Butwal":       4,
    "Bhaktapur":    5,
    "Dhangadhi":    3,
    "Nepalgunj":    3,
    "Itahari":      3,
    "Janakpur":     3,
    "Gorkha":       2,
    "Damak":        2,
    "Lahan":        2,
    "Tulsipur":     2,
    "Siddharthanagar": 2,
    "Baglung":      1,
    "Tansen":       1,
    "Banepa":       1,
    "Dhulikhel":    1,
    "Kirtipur":     1,
    "Mechinagar":   1,
}

cities  = list(NEPAL_CITIES.keys())
weights = np.array(list(NEPAL_CITIES.values()), dtype=float)
weights /= weights.sum()

df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} rows")
print(f"\nMBTI type counts:\n{df['mbti_type'].value_counts().to_string()}")

# Assign locations per MBTI type using the same city distribution
# This ensures location carries zero signal about MBTI type
np.random.seed(42)
df['location'] = ""

for mbti_type, group in df.groupby('mbti_type'):
    n = len(group)
    assigned = np.random.choice(cities, size=n, p=weights)
    df.loc[group.index, 'location'] = assigned

# Verification — check location spread is similar across types
print("\nLocation distribution per MBTI type (top 3 cities shown):")
print("-" * 55)
for mbti_type in sorted(df['mbti_type'].unique()):
    subset = df[df['mbti_type'] == mbti_type]['location']
    top3   = subset.value_counts().head(3)
    top3_str = ", ".join(f"{c}:{v}" for c, v in top3.items())
    print(f"  {mbti_type:<6} (n={len(subset):>3})  {top3_str}")

print(f"\nOverall location distribution (top 10):")
print(df['location'].value_counts().head(10).to_string())

df.to_excel(DATA_PATH, index=False)
print(f"\nSaved → {DATA_PATH}")