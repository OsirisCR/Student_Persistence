"""
Student Persistence Predictor — FastAPI Backend (v2 — fixed feature alignment)
===============================================================================
How to run:
    You need to be sure to have a Python Version supported by TensorFlow 2.12 (e.g. Python 3.10 or 3.12) and then install the dependencies:
    pip install scikit-learn==1.6.1
    pip install fastapi uvicorn tensorflow joblib numpy
    uvicorn predict:app --reload --port 8000
    or, if you're running from the backend directory:
    uvicorn backend.predict:app --reload --port 8000

Visit http://localhost:8000 to verify the API is running.
Visit http://localhost:8000/docs to test it interactively.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib, json, os
import tensorflow as tf

# ---------------------------------------------------------------------------
app = FastAPI(title="Student Persistence Predictor", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["POST", "GET"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
print(f"\nLoading artefacts from: {MODELS_DIR}")

model       = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_model.keras"))
scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
ohe         = joblib.load(os.path.join(MODELS_DIR, "ohe_funding.pkl"))
knn_imputer = joblib.load(os.path.join(MODELS_DIR, "knn_imputer.pkl"))
gw_medians  = joblib.load(os.path.join(MODELS_DIR, "group_medians.pkl"))
train_modes = joblib.load(os.path.join(MODELS_DIR, "train_modes.pkl"))

with open(os.path.join(MODELS_DIR, "pipeline_config.json")) as f:
    config = json.load(f)

TUNED_THRESHOLD = config["tuned_threshold"]
FEATURE_NAMES   = config["feature_names"]       # exact list from X_train at KNN fit time
OHE_COLS        = list(ohe.get_feature_names_out(["funding"]))
SCALE_COLS      = config["scale_cols"]
BINARY_RECODE   = config["binary_recode_cols"]
VALID_RANGES    = config.get("valid_ranges", {})

print(f"  Threshold        : {TUNED_THRESHOLD}")
print(f"  KNN expects      : {knn_imputer.n_features_in_} features")
print(f"  Config features  : {len(FEATURE_NAMES)}")
print(f"  Feature list     : {FEATURE_NAMES}")
print(f"  OHE cols (actual): {OHE_COLS}\n")

# ---------------------------------------------------------------------------
class StudentInput(BaseModel):
    first_term_gpa:  float         = Field(..., ge=0.0, le=4.5)
    second_term_gpa: float         = Field(..., ge=0.0, le=4.5)
    first_language:  int           = Field(..., ge=1, le=2)
    funding:         int           = Field(..., ge=1, le=9)
    fast_track:      int           = Field(..., ge=1, le=2)
    coop:            int           = Field(..., ge=1, le=2)
    residency:       int           = Field(..., ge=1, le=2)
    gender:          int           = Field(..., ge=1, le=2)
    prev_education:  int           = Field(..., ge=1, le=2)
    age_group:       int           = Field(..., ge=1, le=10)
    hs_avg:          float | None  = Field(None, ge=0.0, le=100.0)
    math_score:      float | None  = Field(None, ge=0.0, le=50.0)
    english_grade:   int           = Field(..., ge=1, le=11)

class PredictionOutput(BaseModel):
    persistence:    int
    probability:    float
    confidence_pct: float
    message_title:  str
    message_body:   str
    message_action: str
    message_tier:   str

# ---------------------------------------------------------------------------
def run_pipeline(raw: dict) -> np.ndarray:
    d = raw.copy()

    # 1 — Clip
    for col, bounds in VALID_RANGES.items():
        lo, hi = bounds[0], bounds[1]
        if col in d and d[col] is not None:
            d[col] = max(float(lo), min(float(hi), float(d[col])))

    # 2 — Transforms
    if d.get("first_language") == 3:
        d["first_language"] = 2
    if d.get("first_term_gpa") == 0.0 and (d.get("second_term_gpa") or 0) != 0.0:
        d["first_term_gpa"] = None

    # 3 — hs_avg missing flag
    hs = d.get("hs_avg")
    d["hs_avg_miss_flag"] = 1 if (hs is None or (isinstance(hs, float) and np.isnan(hs))) else 0

    # 4 — Impute first_language from residency
    if d.get("first_language") is None:
        d["first_language"] = 1 if d["residency"] == 1 else 2

    # 5 — Group-wise median imputation
    for col in ["first_term_gpa", "second_term_gpa", "math_score"]:
        val = d.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            d[col] = float(gw_medians[col][1])

    # 6 — Mode imputation for categoricals
    for col in ["gender", "prev_education", "age_group", "english_grade"]:
        if d.get(col) is None:
            d[col] = int(train_modes[col])

    # 7 — Binary recode 1/2 -> 0/1
    for col in BINARY_RECODE:
        if col in d and d[col] is not None:
            d[col] = int(d[col]) - 1

    # 8 — OHE funding
    funding_encoded = ohe.transform([[d["funding"]]])[0]
    for col_name, val in zip(OHE_COLS, funding_encoded):
        d[col_name] = float(val)

    # 9 — StandardScaler
    scale_vals = np.array(
        [[d.get(c) if d.get(c) is not None else np.nan for c in SCALE_COLS]],
        dtype=np.float64
    )
    scaled = scaler.transform(scale_vals)[0]
    for i, col in enumerate(SCALE_COLS):
        d[col] = float(scaled[i])

    # Build feature list without program_completion
    FEATURE_NAMES_NO_PC = [f for f in FEATURE_NAMES if f != "program_completion"]

    # 10 — build row
    row = np.array([[float(d.get(f, 0.0)) for f in FEATURE_NAMES_NO_PC]], dtype=np.float32)

    # 11 — KNN
    row = knn_imputer.transform(row)

    # 12 — compute threshold (same as training)
    gi1 = SCALE_COLS.index("first_term_gpa")
    gi2 = SCALE_COLS.index("second_term_gpa")

    threshold_scaled_1 = (2.0 - scaler.mean_[gi1]) / scaler.scale_[gi1]
    threshold_scaled_2 = (2.0 - scaler.mean_[gi2]) / scaler.scale_[gi2]
    thresh = (threshold_scaled_1 + threshold_scaled_2) / 2

    # 13 — compute program_completion
    idx1 = FEATURE_NAMES_NO_PC.index("first_term_gpa")
    idx2 = FEATURE_NAMES_NO_PC.index("second_term_gpa")

    gpa1 = row[0][idx1]
    gpa2 = row[0][idx2]

    program_completion = 1 if (gpa1 + gpa2)/2 >= thresh else 0

    # 14 — append
    row = np.append(row, [[program_completion]], axis=1)

    print("Final row shape:", row.shape)

    print("KNN expects:", knn_imputer.n_features_in_)

    return row


def build_message(probability: float) -> dict:
    pct = round(probability * 100, 1)
    if probability >= 0.75:
        return {
            "tier"  : "high",
            "title" : "You're on a great path!",
            "body"  : (f"Our model sees a {pct}% likelihood that you will successfully complete "
                       "your first year. Your academic profile shows real strength. Keep showing "
                       "up, keep asking questions, and trust the process — you've got this!"),
            "action": ("Stay connected with your program advisor to make the most of every "
                       "opportunity. Consider joining a student club or study group to keep "
                       "that momentum going."),
        }
    elif probability >= 0.45:
        return {
            "tier"  : "medium",
            "title" : "You're on the right track — with some support you can thrive.",
            "body"  : (f"Our model estimates a {pct}% chance of completing your first year. "
                       "Many students in your situation go on to do brilliantly — the key is "
                       "connecting early with the right support."),
            "action": ("Book a free appointment with Academic Advising at "
                       "continuingeducation@centennialcollege.ca — they are there specifically "
                       "to help you succeed."),
        }
    else:
        return {
            "tier"  : "low",
            "title" : "Let's make sure you have everything you need.",
            "body"  : (f"Our model flags some potential challenges ahead ({pct}% persistence "
                       "likelihood). This is NOT a judgment — it is an early signal so we can "
                       "get you the right support before small obstacles become big ones. "
                       "Many students who started just like you went on to graduate with honours."),
            "action": ("Please reach out to Student Services today: "
                       "continuingeducation@centennialcollege.ca or visit the Student Success "
                       "Centre on campus. You are not alone — the earlier you connect, the "
                       "better the outcome."),
        }

# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status"           : "ok",
        "features_expected": len(FEATURE_NAMES),
        "ohe_cols_actual"  : OHE_COLS,
        "feature_names"    : FEATURE_NAMES,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    try:
        X    = run_pipeline(student.model_dump())
        prob = float(model.predict(X, verbose=0)[0][0])
        msg  = build_message(prob)
        return PredictionOutput(
            persistence    = 1 if prob >= TUNED_THRESHOLD else 0,
            probability    = round(prob, 4),
            confidence_pct = round(prob * 100, 1),
            message_title  = msg["title"],
            message_body   = msg["body"],
            message_action = msg["action"],
            message_tier   = msg["tier"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")