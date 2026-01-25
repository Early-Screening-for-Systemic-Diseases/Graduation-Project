import os
import joblib
from fastapi import FastAPI
import pandas as pd

# Load the large model from HuggingFace download location in Docker
if os.path.exists("/app/diabetes_survey_model.pkl"):
    model_path = "/app/diabetes_survey_model.pkl"
else:
    # Fallback for local testing
    model_path = r"D:\Survey_T_Dataset\diabetes_survey_model.pkl"

# Load the small columns file from GitHub / local repo
columns_path = "model_columns.pkl"  # this will be copied with your code

# Load the model and columns
model = joblib.load(model_path)
columns = joblib.load(columns_path)

print(f"Model loaded from: {model_path}")
print(f"Columns loaded from: {columns_path}")




app = FastAPI()

# Load model
model = joblib.load("diabetes_survey_model.pkl")
columns = joblib.load("model_columns.pkl")

@app.get("/")
def home():
    return {"message": "Diabetes Survey API Running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    pred = 1 if prob > 0.4 else 0

    return {
        "diabetes": "Yes" if pred == 1 else "No",
        "probability": float(prob)
    }
