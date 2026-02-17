from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the model (already pushed to Railway repo)
model_path = "diabetes_gb_balanced_model.pkl"
model = joblib.load(model_path)

print(f"Model loaded from: {model_path}")

# ================= API ENDPOINTS =================

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
        "PhysActivity", 0
        "HeartDiseaseorAttack 0",
        "Stroke" 0:


      
    }
    """

    df = pd.DataFrame([data])

    # Predict
    pred_class = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    message = f"Likely to have diabetes ({prob:.1%})" if pred_class == 1 else f"Not likely to have diabetes ({1-prob:.1%})"

    return {
        "prediction": int(pred_class),
        "probability": float(prob),
        "message": message
    }