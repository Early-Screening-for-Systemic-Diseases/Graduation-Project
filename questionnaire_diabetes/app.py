import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Diabetes Survey API")

MODEL_FILENAME = "diabetes_gb_balanced_model.pkl"
COLUMNS_FILENAME = "model_columns.pkl"

# locate model in common locations
candidate_paths = [
    os.path.join("/app", MODEL_FILENAME),
    os.path.join(os.getcwd(), MODEL_FILENAME),
    MODEL_FILENAME,
]

model = None
model_path = None
for p in candidate_paths:
    if os.path.exists(p):
        model = joblib.load(p)
        model_path = p
        break

if model is None:
    raise RuntimeError(f"Model not found. Place {MODEL_FILENAME} in the service folder before deploying.")

# try load columns file
columns = None
for p in [os.path.join("/app", COLUMNS_FILENAME), os.path.join(os.getcwd(), COLUMNS_FILENAME), COLUMNS_FILENAME]:
    if os.path.exists(p):
        columns = joblib.load(p)
        break

if columns is None:
    if hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
    elif hasattr(model, "n_features_in_"):
        n = int(model.n_features_in_)
        columns = [f"f{i}" for i in range(n)]
    else:
        columns = None

class InputPayload(BaseModel):
    data: Dict[str, Any]


@app.get("/")
def home():
    return {"message": "Diabetes Survey API Running 🚀"}


@app.post("/predict")
def predict(payload: InputPayload):
    df = pd.DataFrame([payload.data])
    df = pd.get_dummies(df)

    if columns is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model feature columns not available. Provide model_columns.pkl (pickled list of columns), "
                "or ensure the trained model exposes feature names."
            ),
        )

    df = df.reindex(columns=columns, fill_value=0)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[0][1])
    else:
        pred = int(model.predict(df)[0])
        prob = 1.0 if pred == 1 else 0.0

    pred = 1 if prob > 0.4 else 0

    return {"diabetes": "Yes" if pred == 1 else "No", "probability": prob}
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Diabetes Survey API")

MODEL_FILENAME = "diabetes_gb_balanced_model.pkl"
COLUMNS_FILENAME = "model_columns.pkl"

# locate model
candidate_paths = [
    os.path.join("/app", MODEL_FILENAME),
    os.path.join(os.getcwd(), MODEL_FILENAME),
    MODEL_FILENAME
]

model = None
for p in candidate_paths:
    if os.path.exists(p):
        model = joblib.load(p)
        model_path = p
        break

if model is None:
    raise RuntimeError(f"Model not found. Place {MODEL_FILENAME} in the service folder before deploying.")

# load columns if present
columns = None
for p in [os.path.join("/app", COLUMNS_FILENAME), os.path.join(os.getcwd(), COLUMNS_FILENAME), COLUMNS_FILENAME]:
    if os.path.exists(p):
        columns = joblib.load(p)
        break

if columns is None:
    if hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
    elif hasattr(model, "n_features_in_"):
        n = int(model.n_features_in_)
        columns = [f"f{i}" for i in range(n)]
    else:
        columns = None

class InputPayload(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def home():
    return {"message": "Diabetes Survey API Running 🚀"}

@app.post("/predict")
def predict(payload: InputPayload):
    df = pd.DataFrame([payload.data])
    df = pd.get_dummies(df)

    if columns is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model feature columns not available. Provide model_columns.pkl (pickled list of columns), "
                "or ensure the trained model exposes feature names."
            )
        )

    df = df.reindex(columns=columns, fill_value=0)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[0][1])
    else:
        pred = int(model.predict(df)[0])
        prob = 1.0 if pred == 1 else 0.0

    pred = 1 if prob > 0.4 else 0

    return {"diabetes": "Yes" if pred == 1 else "No", "probability": prob}