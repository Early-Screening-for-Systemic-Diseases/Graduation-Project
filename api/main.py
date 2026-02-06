from fastapi import FastAPI
from nlp.shared.predictor import predict  # Import your existing predictor

app = FastAPI(title="NLP Disease Prediction API")

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/predict")
def run_prediction(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "Text is required"}
    return predict(text)
