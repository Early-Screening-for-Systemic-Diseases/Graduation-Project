from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="Anemia Questionnaire API")

# =========================
# Load model ONCE
# =========================
MODEL_PATH = "anemia_questionnaire_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# Feature order (CRITICAL)
# =========================
FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",

    "DBQ197",
    "DBQ235A",
    "DBQ229",
    "DBQ223A",

    "RHQ010",
    "RHQ020",
    "RHQ060",

    "MCQ010",
    "MCQ053",
    "MCQ080",
    "MCQ160A",
    "MCQ025",
    "MCQ040",

    "LBXHGB"
]

# =========================
# Request Schema
# =========================
class AnemiaRequest(BaseModel):
    RIDAGEYR: float
    RIAGENDR: int
    RIDRETH1: int

    DBQ197: float
    DBQ235A: float
    DBQ229: float
    DBQ223A: float

    RHQ010: float
    RHQ020: float
    RHQ060: float

    MCQ010: float
    MCQ053: float
    MCQ080: float
    MCQ160A: float
    MCQ025: float
    MCQ040: float

    LBXHGB: float

# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict_anemia(data: AnemiaRequest):

    try:
        input_data = [getattr(data, f) for f in FEATURES]
        X = np.array(input_data).reshape(1, -1)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return {
            "anemia_prediction": int(prediction),
            "label": "Anemic" if prediction == 1 else "Non-Anemic",
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
