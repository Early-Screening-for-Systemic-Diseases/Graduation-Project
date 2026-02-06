from fastapi import FastAPI
from nlp.shared.predictor import predict  # Import your existing predictor
import subprocess
import sys

def install_runtime_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--no-cache-dir", "torch==2.10.0", 
                           "sentence-transformers==5.2.2",
                           "spacy==3.8.11"])
    import spacy
    spacy.load("en_core_web_sm")

install_runtime_packages()

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
