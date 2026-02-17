# app.py
from fastapi import FastAPI
import joblib
import pandas as pd
import os

# Create FastAPI instance
app = FastAPI()

# Load the model
model_path = "diabetes_gb_balanced_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)
print(f"Model loaded from: {model_path}")

# ======= API ENDPOINTS =======

@app.get("/")
def home():
    return {"message": "Diabetes Survey API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    """
    Input JSON should include all required survey features.
    Example:
    {
        "GenHlth": 3,
        "BMI": 31,
        "HighBP": 1,
        "Age": 9,
        "HighChol": 1,
        "PhysHlth": 5,
        "DiffWalk": 0,
        "PhysActivity": 0,
        "HeartDiseaseorAttack": 0,
        "Stroke": 0
    }
    """

    # Convert JSON to DataFrame
    df = pd.DataFrame([data])

    # Predict class and probability
    pred_class = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    message = (
        f"Likely to have diabetes ({prob:.1%})"
        if pred_class == 1
        else f"Not likely to have diabetes ({(1-prob):.1%})"
    )

    return {
        "prediction": pred_class,
        "probability": prob,
        "message": message
    }