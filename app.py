import pandas as pd
import re
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from scipy.special import expit

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DATA_PATH = r"C:\Users\NITRO\Documents\mbti__1.xlsx"

# All text columns — updated to match the full dataset schema
TEXT_COLUMNS = [
    'profession', 'hobbies', 'interests', 'bio',
    'personality_traits', 'lifestyle_habits',
    'values_beliefs', 'goals_ambitions', 'fun_fact',
    # New columns
    'social_preference', 'skills_talents',
    'relationship_goal', 'communication_style',
    'horoscope_sign', 'location',
]

# Non-text / metadata columns (used for output/display only, not for training)
META_COLUMNS = ['location', 'birthdate', 'age', 'horoscope_sign',
                'relationship_goal', 'communication_style']

VECTOR_CONFIGS = [
    {'max_features': 7000,  'ngram_range': (1, 2), 'stop_words': 'english', 'min_df': 2},
    {'max_features': 10000, 'ngram_range': (1, 2), 'stop_words': 'english', 'min_df': 2},
    {'max_features': 12000, 'ngram_range': (1, 3), 'stop_words': 'english', 'min_df': 2},
    {'max_features': 15000, 'ngram_range': (1, 3), 'stop_words': 'english', 'min_df': 3},
]

CANDIDATE_MODELS = {
    "LinearSVC":          LinearSVC(C=1.0, max_iter=2000, random_state=42),
    "LinearSVC_C0.5":     LinearSVC(C=0.5, max_iter=2000, random_state=42),
    "LinearSVC_C0.1":     LinearSVC(C=0.1, max_iter=2000, random_state=42),
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42),
    "ComplementNB":       ComplementNB(alpha=0.1),
    "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
}

TRAITS       = ['IE', 'NS', 'TF', 'JP']
TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5
MODEL_PATH   = "mbti_model.pkl"

# Validation config
MIN_WORDS_STRICT = 10
MIN_WORDS_WARN   = 20
MAX_CHARS        = 5000
MIN_UNIQUE_WORDS = 6

SPAM_PATTERNS = [
    r"(.)\1{6,}",
    r"(\b\w+\b)(\s+\1){4,}",
    r"^[^a-zA-Z]*$",
]


# ──────────────────────────────────────────────
# TEXT HELPERS
# ──────────────────────────────────────────────

def light_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def combine_profile(row) -> str:
    """Combine all available text columns into a single profile string."""
    parts = []
    for col in TEXT_COLUMNS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def safe_int(val, default=None):
    """Safely convert a value to int."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def parse_birthdate(val):
    """Parse birthdate string or datetime to a date string 'YYYY-MM-DD'."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip()
        return val if val else None
    try:
        return pd.Timestamp(val).strftime('%Y-%m-%d')
    except Exception:
        return None


# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────

def validate_bio(bio):
    if not isinstance(bio, str):
        return False, "INVALID_TYPE", "Bio must be a string."
    bio = bio.strip()
    if not bio:
        return False, "EMPTY", "Bio cannot be empty."
    if len(bio) > MAX_CHARS:
        return False, "TOO_LONG", f"Bio is too long ({len(bio)} chars). Keep it under {MAX_CHARS} characters."
    words      = bio.split()
    word_count = len(words)
    if word_count < MIN_WORDS_STRICT:
        return False, "TOO_SHORT", f"Bio is too short ({word_count} words). Please provide at least {MIN_WORDS_STRICT} words."
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, bio, re.IGNORECASE):
            return False, "SPAM", "Bio appears to contain repeated or invalid content."
    unique_words = len(set(w.lower() for w in words))
    if unique_words < MIN_UNIQUE_WORDS:
        return False, "LOW_DIVERSITY", f"Bio has too few unique words ({unique_words}). Write a more descriptive profile."
    alpha_ratio = sum(c.isalpha() for c in bio) / max(len(bio), 1)
    if alpha_ratio < 0.5:
        return False, "NOT_TEXT", "Bio contains too many non-letter characters."
    return True, "OK", "Valid"


def get_warning(bio):
    word_count = len(bio.split())
    if word_count < MIN_WORDS_WARN:
        return f"Bio is short ({word_count} words). Predictions are more reliable with 20+ words."
    return None


# ──────────────────────────────────────────────
# STEP 1 — FIND BEST TF-IDF CONFIG
# ──────────────────────────────────────────────

def find_best_config(df):
    print("\nStep 1: Finding best TF-IDF config...")
    print("-" * 55)

    best_config = None
    best_avg    = 0

    for config in VECTOR_CONFIGS:
        tr, te = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=df['mbti_type']
        )
        vec     = TfidfVectorizer(**config)
        X_train = vec.fit_transform(tr['profile_text'])
        X_test  = vec.transform(te['profile_text'])

        accs = []
        for trait in TRAITS:
            clf = LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
            clf.fit(X_train, tr[trait])
            accs.append(accuracy_score(te[trait], clf.predict(X_test)))

        avg = np.mean(accs)
        print(f"  max_features={config['max_features']:>5}  "
              f"ngram={config['ngram_range']}  "
              f"avg_trait_acc={avg:.2%}")

        if avg > best_avg:
            best_avg    = avg
            best_config = config

    print(f"\n  Selected: max_features={best_config['max_features']}  "
          f"ngram={best_config['ngram_range']}")
    return best_config


# ──────────────────────────────────────────────
# STEP 2 — COMPARE ALL MODELS WITH CV
# ──────────────────────────────────────────────

def run_cv_for_model(df, model_template, tfidf_config, n_splits=CV_FOLDS):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(df['profile_text'], df['mbti_type']))

    trait_scores = {t: [] for t in TRAITS}
    mbti_scores  = []

    for train_idx, test_idx in splits:
        train_texts = df['profile_text'].iloc[train_idx]
        test_texts  = df['profile_text'].iloc[test_idx]

        vec     = TfidfVectorizer(**tfidf_config)
        X_train = vec.fit_transform(train_texts)
        X_test  = vec.transform(test_texts)

        fold_preds = {}
        for trait in TRAITS:
            y_train = df[trait].iloc[train_idx].values
            y_test  = df[trait].iloc[test_idx].values
            clf     = clone(model_template)
            clf.fit(X_train, y_train)
            preds   = clf.predict(X_test)
            trait_scores[trait].append(accuracy_score(y_test, preds))
            fold_preds[trait] = (preds, y_test)

        n         = len(test_idx)
        mbti_pred = ["".join(fold_preds[t][0][i] for t in TRAITS) for i in range(n)]
        mbti_true = ["".join(fold_preds[t][1][i] for t in TRAITS) for i in range(n)]
        mbti_scores.append(accuracy_score(mbti_true, mbti_pred))

    return {
        "mbti_mean":  round(np.mean(mbti_scores) * 100, 2),
        "mbti_std":   round(np.std(mbti_scores)  * 100, 2),
        "trait_mean": round(np.mean([np.mean(trait_scores[t]) for t in TRAITS]) * 100, 2),
        "per_trait":  {t: round(np.mean(trait_scores[t]) * 100, 2) for t in TRAITS},
    }


def compare_models(df, best_config):
    print(f"\nStep 2: Comparing {len(CANDIDATE_MODELS)} models ({CV_FOLDS}-fold CV)...")
    print("-" * 55)

    results = []
    for name, model_template in CANDIDATE_MODELS.items():
        print(f"  Testing {name:<28} ", end="", flush=True)
        scores          = run_cv_for_model(df, model_template, best_config)
        scores['model'] = name
        results.append(scores)
        print(f"MBTI: {scores['mbti_mean']:.2f}% ±{scores['mbti_std']:.2f}%")

    results_sorted = sorted(results, key=lambda x: x['mbti_mean'], reverse=True)

    print("\n" + "=" * 74)
    print(f"  MODEL COMPARISON  ({CV_FOLDS}-fold CV)")
    print("=" * 74)
    print(f"  {'Rank':<5} {'Model':<26} {'MBTI%':>7} {'±Std':>6} "
          f"{'Trait%':>7}  IE     NS     TF     JP")
    print("-" * 74)

    for i, r in enumerate(results_sorted):
        pt     = r['per_trait']
        marker = "  ◄ BEST" if i == 0 else ""
        print(
            f"  {i+1:<5} {r['model']:<26} {r['mbti_mean']:>6.2f}%"
            f"  ±{r['mbti_std']:.2f}%  {r['trait_mean']:>6.2f}%"
            f"  {pt['IE']:.1f}  {pt['NS']:.1f}  {pt['TF']:.1f}  {pt['JP']:.1f}"
            f"{marker}"
        )

    print("=" * 74)
    best          = results_sorted[0]
    weakest_trait = min(best['per_trait'], key=best['per_trait'].get)
    print(f"\n  Best model    : {best['model']}")
    print(f"  MBTI accuracy : {best['mbti_mean']:.2f}% ± {best['mbti_std']:.2f}%")
    print(f"  Trait accuracy: {best['trait_mean']:.2f}%")
    print(f"  Weakest trait : {weakest_trait} "
          f"({best['per_trait'][weakest_trait]:.1f}%) — most room to improve")
    return best['model']


# ──────────────────────────────────────────────
# STEP 3 — DETAILED CV ON BEST MODEL
# ──────────────────────────────────────────────

def run_cross_validation(df, config, model_template, n_splits=CV_FOLDS):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(df['profile_text'], df['mbti_type']))

    trait_scores = {t: [] for t in TRAITS}
    mbti_scores  = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/{n_splits} ", end="", flush=True)

        train_texts = df['profile_text'].iloc[train_idx]
        test_texts  = df['profile_text'].iloc[test_idx]

        vec     = TfidfVectorizer(**config)
        X_train = vec.fit_transform(train_texts)
        X_test  = vec.transform(test_texts)

        fold_preds = {}
        for trait in TRAITS:
            y_train = df[trait].iloc[train_idx].values
            y_test  = df[trait].iloc[test_idx].values
            clf     = clone(model_template)
            clf.fit(X_train, y_train)
            preds   = clf.predict(X_test)
            trait_scores[trait].append(accuracy_score(y_test, preds))
            fold_preds[trait] = (preds, y_test)

        n         = len(test_idx)
        mbti_pred = ["".join(fold_preds[t][0][i] for t in TRAITS) for i in range(n)]
        mbti_true = ["".join(fold_preds[t][1][i] for t in TRAITS) for i in range(n)]
        mbti_acc  = accuracy_score(mbti_true, mbti_pred)
        mbti_scores.append(mbti_acc)
        print(f"| MBTI: {mbti_acc:.2%}", flush=True)

    return trait_scores, mbti_scores


def print_cv_report(trait_scores, mbti_scores):
    print("\n" + "=" * 52)
    print(f"  FINAL CV REPORT  ({CV_FOLDS}-fold)")
    print("=" * 52)
    print(f"  {'Trait':<6}  {'Mean':>7}  {'±Std':>6}  {'Min':>7}  {'Max':>7}")
    print("-" * 52)
    for trait in TRAITS:
        s = trait_scores[trait]
        print(f"  {trait:<6}  {np.mean(s):>7.2%}  ±{np.std(s):.2%}  "
              f"{min(s):>7.2%}  {max(s):>7.2%}")
    print("-" * 52)
    print(f"  {'MBTI':<6}  {np.mean(mbti_scores):>7.2%}  ±{np.std(mbti_scores):.2%}  "
          f"{min(mbti_scores):>7.2%}  {max(mbti_scores):>7.2%}")
    print("=" * 52)
    std = np.std(mbti_scores)
    if std < 0.01:
        print("\n  Stability: EXCELLENT — consistent across all folds.")
    elif std < 0.02:
        print("\n  Stability: GOOD — minor variance across folds.")
    else:
        print("\n  Stability: REVIEW — high variance; consider more data.")
    print()


# ──────────────────────────────────────────────
# STEP 4 — TRAIN FINAL MODEL ON ALL DATA
# ──────────────────────────────────────────────

def train_final_model(df, config, best_model_name):
    print(f"\nStep 4: Training final [{best_model_name}] on all {len(df)} rows...")

    model_template = CANDIDATE_MODELS[best_model_name]
    vec            = TfidfVectorizer(**config)
    X_all          = vec.fit_transform(df['profile_text'])

    models   = {}
    encoders = {}

    for trait in TRAITS:
        le = LabelEncoder()
        y  = le.fit_transform(df[trait])
        clf = clone(model_template)
        clf.fit(X_all, y)
        models[trait]   = clf
        encoders[trait] = le

    bundle = {
        'vectorizer': vec,
        'models':     models,
        'encoders':   encoders,
        'config':     config,
        'model_name': best_model_name,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"  Saved → {MODEL_PATH}")
    return vec, models, encoders


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────

def predict(text, vectorizer, models, encoders):
    cleaned = light_clean(text)
    X       = vectorizer.transform([cleaned])

    letters     = []
    confidences = {}

    for trait in TRAITS:
        clf      = models[trait]
        le       = encoders[trait]
        raw_pred = clf.predict(X)[0]
        letter   = le.inverse_transform([raw_pred])[0]
        letters.append(letter)

        try:
            decision = clf.decision_function(X)[0]
            prob     = float(expit(decision))
            conf     = prob if raw_pred == 1 else (1 - prob)
        except AttributeError:
            try:
                proba = clf.predict_proba(X)[0]
                conf  = float(max(proba))
            except AttributeError:
                conf  = 1.0

        confidences[trait] = (letter, round(conf * 100, 1))

    return "".join(letters), confidences


def explain(text, vectorizer, models, encoders):
    """
    Returns top 5 words from the bio that most influenced each trait prediction.
    Only works for LinearSVC (uses coef_ weights).
    Falls back to empty list for other model types.
    """
    cleaned       = light_clean(text)
    X             = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    X_arr         = X.toarray()[0]

    present_idx = np.where(X_arr > 0)[0]

    explanations = {}

    for trait in TRAITS:
        clf      = models[trait]
        le       = encoders[trait]
        raw_pred = clf.predict(X)[0]

        if not hasattr(clf, 'coef_'):
            explanations[trait] = []
            continue

        coef      = clf.coef_[0]
        direction = 1 if raw_pred == 1 else -1

        scored = [
            (feature_names[i], float(direction * coef[i] * X_arr[i]))
            for i in present_idx
        ]

        top = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
        top = [(w, round(s, 4)) for w, s in top if s > 0]

        explanations[trait] = top

    return explanations


def print_prediction(mbti, confidences, explanations=None):
    print(f"\n  Predicted MBTI: {mbti}")
    print("  Trait breakdown:")
    for trait, (letter, conf) in confidences.items():
        filled = int(conf / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"    {trait}: {letter}  {bar}  {conf:.1f}%")
        if explanations and explanations.get(trait):
            words = ", ".join(w for w, _ in explanations[trait])
            print(f"         key words: {words}")
    print()


# ──────────────────────────────────────────────
# LOAD SAVED MODEL
# ──────────────────────────────────────────────

def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"  Loaded model: {bundle.get('model_name', 'unknown')}")
    return bundle['vectorizer'], bundle['models'], bundle['encoders']


# ──────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

flask_app = Flask(__name__, static_folder=".")
CORS(flask_app)

_vectorizer = None
_models     = None
_encoders   = None
MODEL_READY = False


@flask_app.route("/")
def index():
    return send_from_directory(".", "index.html")


@flask_app.route("/predict", methods=["POST"])
def predict_route():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded.", "code": "MODEL_NOT_FOUND"}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON.", "code": "INVALID_JSON"}), 400

    # ── Build profile text from all available fields ──────────────────────────
    # If a full profile dict is sent, combine all text fields.
    # If only "bio" is sent (legacy), fall back to that.
    profile_fields = {col: data.get(col, "") for col in TEXT_COLUMNS}
    profile_text   = " ".join(
        str(v).strip() for v in profile_fields.values() if v and str(v).strip()
    )

    # Fallback: if nothing useful came from profile fields, try bare "bio"
    if not profile_text.strip():
        profile_text = data.get("bio", "")

    # Validate using the combined text (treated as "bio" for validation purposes)
    is_valid, error_code, message = validate_bio(profile_text)
    if not is_valid:
        return jsonify({"error": message, "code": error_code}), 422

    mbti, confidences = predict(profile_text, _vectorizer, _models, _encoders)
    explanations      = explain(profile_text, _vectorizer, _models, _encoders)

    traits_out = {
        trait: {
            "letter":     letter,
            "confidence": conf,
            "top_words":  explanations.get(trait, []),
        }
        for trait, (letter, conf) in confidences.items()
    }

    # ── Include parsed metadata in response ───────────────────────────────────
    meta = {
        "location":           data.get("location") or None,
        "age":                safe_int(data.get("age")),
        "birthdate":          parse_birthdate(data.get("birthdate")),
        "horoscope_sign":     data.get("horoscope_sign") or None,
        "relationship_goal":  data.get("relationship_goal") or None,
        "communication_style": data.get("communication_style") or None,
        "social_preference":  data.get("social_preference") or None,
    }

    return jsonify({
        "type":    mbti,
        "traits":  traits_out,
        "warning": get_warning(profile_text),
        "meta":    meta,
    }), 200


@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if MODEL_READY else "degraded",
                    "model_loaded": MODEL_READY}), 200


# ──────────────────────────────────────────────
# MAIN
#
#   python app.py          → train + save model, then start web server
#   python app.py --train  → train + save model only (no server)
#   python app.py --serve  → load saved model and start web server only
# ──────────────────────────────────────────────

def main():
    global _vectorizer, _models, _encoders, MODEL_READY

    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    # ── SERVE ONLY ─────────────────────────────
    if mode == "--serve":
        print("Loading saved model...")
        try:
            _vectorizer, _models, _encoders = load_model()
            MODEL_READY = True
        except FileNotFoundError:
            print(f"  ERROR: {MODEL_PATH} not found. Run 'python app.py --train' first.")
            return
        print("Starting web server → http://localhost:5000")
        flask_app.run(debug=False, port=5000)
        return

    # ── TRAIN ──────────────────────────────────
    print("Loading dataset...")
    df = pd.read_excel(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Normalise column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print("\nMBTI Distribution:")
    print(df['mbti_type'].value_counts().to_string())

    # Derive per-trait labels from mbti_type
    df['IE'] = df['mbti_type'].str[0]
    df['NS'] = df['mbti_type'].str[1]
    df['TF'] = df['mbti_type'].str[2]
    df['JP'] = df['mbti_type'].str[3]

    # Report which TEXT_COLUMNS are present vs. missing
    present_cols = [c for c in TEXT_COLUMNS if c in df.columns]
    missing_cols = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"\n  Note: These TEXT_COLUMNS are not in the dataset and will be skipped: {missing_cols}")

    print("\nCombining text fields...")
    df['profile_text'] = df.apply(combine_profile, axis=1).apply(light_clean)
    df = df[df['profile_text'].str.strip() != ""].copy()
    print(f"Rows after cleaning: {len(df)}")

    best_config     = find_best_config(df)
    best_model_name = compare_models(df, best_config)

    print(f"\nStep 3: Detailed CV report for [{best_model_name}]...")
    print("-" * 52)
    trait_scores, mbti_scores = run_cross_validation(
        df, best_config, CANDIDATE_MODELS[best_model_name]
    )
    print_cv_report(trait_scores, mbti_scores)

    _vectorizer, _models, _encoders = train_final_model(df, best_config, best_model_name)
    MODEL_READY = True

    # ── TRAIN ONLY ─────────────────────────────
    if mode == "--train":
        print("\nTraining complete. Run 'python app.py --serve' to start the web server.")
        return

    # ── DEFAULT: train then start server ───────
    print("\nStarting web server → http://localhost:5000")
    flask_app.run(debug=False, port=5000)


if __name__ == "__main__":
    main()