from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# =========================
# Load model
# =========================
model = joblib.load("anemia_questionnaire_model.pkl")

THRESHOLD = 0.4

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Anemia Questionnaire API",
    description="Predict anemia risk using questionnaire data",
    version="1.0"
)

# =========================
# Input Schema
# =========================
class QuestionnaireInput(BaseModel):
    RIDAGEYR: int = Field(..., example=38)
    RIAGENDR: int = Field(..., example=2, description="1=Male, 2=Female")
    RIDRETH1: int = Field(..., example=4)
    MCQ010: int = Field(..., example=2, description="Diabetes: 1=Yes, 2=No")
    MCQ053: int = Field(..., example=1, description="Hypertension: 1=Yes, 2=No")
    MCQ080: int = Field(..., example=2, description="Heart disease: 1=Yes, 2=No")
    MCQ160A: int = Field(..., example=2, description="Asthma: 1=Yes, 2=No")

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"message": "Anemia Questionnaire API is running"}

@app.post("/predict")
def predict_anemia(data: QuestionnaireInput):

    # Convert input to DataFrame (VERY IMPORTANT)
    input_df = pd.DataFrame([data.dict()])

    # Predict probability
    probability = model.predict_proba(input_df)[:, 1][0]

    # Binary prediction
    prediction = int(probability >= THRESHOLD)

    return {
        "anemia_probability": round(float(probability), 4),
        
    }
