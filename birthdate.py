import pandas as pd
import numpy as np
from datetime import date, timedelta

DATA_PATH = r"C:\Users\NITRO\Documents\mbti__1.xlsx"
TODAY     = date.today()

# Age buckets with weights — 20-35 gets ~70% of records
AGE_BUCKETS = [
    (18, 19, 5),
    (20, 35, 70),
    (36, 45, 25),
]

def random_birthdate_weighted(rng):
    # Pick a bucket by weight
    buckets = [(lo, hi) for lo, hi, _ in AGE_BUCKETS]
    weights = np.array([w for _, _, w in AGE_BUCKETS], dtype=float)
    weights /= weights.sum()
    idx     = rng.choice(len(buckets), p=weights)
    lo, hi  = buckets[idx]
    age     = int(rng.integers(lo, hi + 1))
    # Random day within that age's birth year range
    latest   = TODAY - timedelta(days=age * 365)
    earliest = TODAY - timedelta(days=(age + 1) * 365)
    delta    = (latest - earliest).days
    return earliest + timedelta(days=int(rng.integers(0, delta)))

def horoscope(d):
    m, day = d.month, d.day
    if   (m==3  and day>=21) or (m==4  and day<=19): return "Aries"
    elif (m==4  and day>=20) or (m==5  and day<=20): return "Taurus"
    elif (m==5  and day>=21) or (m==6  and day<=20): return "Gemini"
    elif (m==6  and day>=21) or (m==7  and day<=22): return "Cancer"
    elif (m==7  and day>=23) or (m==8  and day<=22): return "Leo"
    elif (m==8  and day>=23) or (m==9  and day<=22): return "Virgo"
    elif (m==9  and day>=23) or (m==10 and day<=22): return "Libra"
    elif (m==10 and day>=23) or (m==11 and day<=21): return "Scorpio"
    elif (m==11 and day>=22) or (m==12 and day<=21): return "Sagittarius"
    elif (m==12 and day>=22) or (m==1  and day<=19): return "Capricorn"
    elif (m==1  and day>=20) or (m==2  and day<=18): return "Aquarius"
    else:                                             return "Pisces"

def calc_age(d):
    return TODAY.year - d.year - ((TODAY.month, TODAY.day) < (d.month, d.day))

df  = pd.read_excel(DATA_PATH)
rng = np.random.default_rng(42)
print(f"Loaded {len(df)} rows")

birthdates = [random_birthdate_weighted(rng) for _ in range(len(df))]

df['birthdate']      = [d.strftime("%Y-%m-%d") for d in birthdates]
df['age']            = [calc_age(d) for d in birthdates]
df['horoscope_sign'] = [horoscope(d) for d in birthdates]

print(f"\nAge distribution:\n{df['age'].value_counts().sort_index().to_string()}")
print(f"\nAge bucket summary:")
print(f"  18-19 : {len(df[df['age'].between(18,19)]):>4} ({len(df[df['age'].between(18,19)])/len(df)*100:.1f}%)")
print(f"  20-35 : {len(df[df['age'].between(20,35)]):>4} ({len(df[df['age'].between(20,35)])/len(df)*100:.1f}%)")
print(f"  36-45 : {len(df[df['age'].between(36,45)]):>4} ({len(df[df['age'].between(36,45)])/len(df)*100:.1f}%)")
print(f"\nHoroscope distribution:\n{df['horoscope_sign'].value_counts().to_string()}")
print(f"\nSample rows:\n{df[['birthdate','age','horoscope_sign']].head(5).to_string()}")

df.to_excel(DATA_PATH, index=False)
print(f"\nSaved -> {DATA_PATH}")