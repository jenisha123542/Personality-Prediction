import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\NITRO\Documents\mbti__1.xlsx"

# Per-type weights: [Texter, Caller, Voice notes, Video caller, In-person only]
# Derived from I/E + T/F + N/S trait logic
STYLE_MAP = {
    #              Texter Caller VoiceN VideoC InPers
    "INTJ":  dict(Texter=45, Caller=10, Voice_notes=20, Video_caller=10, In_person_only=15),
    "INTP":  dict(Texter=50, Caller=10, Voice_notes=15, Video_caller=10, In_person_only=15),
    "INFJ":  dict(Texter=30, Caller=10, Voice_notes=25, Video_caller=10, In_person_only=25),
    "INFP":  dict(Texter=35, Caller=10, Voice_notes=30, Video_caller=10, In_person_only=15),
    "ISTJ":  dict(Texter=35, Caller=30, Voice_notes=10, Video_caller=10, In_person_only=15),
    "ISTP":  dict(Texter=40, Caller=25, Voice_notes=10, Video_caller=10, In_person_only=15),
    "ISFJ":  dict(Texter=30, Caller=20, Voice_notes=20, Video_caller=10, In_person_only=20),
    "ISFP":  dict(Texter=30, Caller=15, Voice_notes=25, Video_caller=10, In_person_only=20),
    "ENTJ":  dict(Texter=20, Caller=35, Voice_notes=10, Video_caller=25, In_person_only=10),
    "ENTP":  dict(Texter=25, Caller=25, Voice_notes=15, Video_caller=20, In_person_only=15),
    "ENFJ":  dict(Texter=15, Caller=25, Voice_notes=20, Video_caller=15, In_person_only=25),
    "ENFP":  dict(Texter=20, Caller=20, Voice_notes=25, Video_caller=15, In_person_only=20),
    "ESTJ":  dict(Texter=15, Caller=40, Voice_notes=10, Video_caller=20, In_person_only=15),
    "ESTP":  dict(Texter=15, Caller=35, Voice_notes=10, Video_caller=20, In_person_only=20),
    "ESFJ":  dict(Texter=15, Caller=30, Voice_notes=20, Video_caller=15, In_person_only=20),
    "ESFP":  dict(Texter=10, Caller=30, Voice_notes=20, Video_caller=25, In_person_only=15),
}

# Normalize label keys back to display names
LABEL_MAP = {
    "Texter": "Texter", "Caller": "Caller",
    "Voice_notes": "Voice notes", "Video_caller": "Video caller",
    "In_person_only": "In-person only"
}

df  = pd.read_excel(DATA_PATH)
rng = np.random.default_rng(42)
print(f"Loaded {len(df)} rows")

df['communication_style'] = ""

for mbti_type, group in df.groupby('mbti_type'):
    if mbti_type not in STYLE_MAP:
        continue
    raw     = STYLE_MAP[mbti_type]
    labels  = [LABEL_MAP[k] for k in raw]
    weights = np.array(list(raw.values()), dtype=float)
    weights /= weights.sum()
    df.loc[group.index, 'communication_style'] = rng.choice(
        labels, size=len(group), p=weights
    )

print(f"\nOverall distribution:\n{df['communication_style'].value_counts().to_string()}")
print(f"\nPer MBTI type:")
print("-" * 65)
for t in sorted(df['mbti_type'].unique()):
    sub = df[df['mbti_type'] == t]['communication_style'].value_counts()
    print(f"  {t:<6} (n={len(df[df['mbti_type']==t]):>3})  " +
          ", ".join(f"{c}:{v}" for c, v in sub.items()))

df.to_excel(DATA_PATH, index=False)
print(f"\nSaved -> {DATA_PATH}")