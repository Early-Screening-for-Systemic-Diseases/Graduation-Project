import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline.pipeline import run_pipeline

app = FastAPI(
    title="NLP Disease Prediction API",
    description="Early Screening for Systemic Diseases",
    version="2.0.0"
)

# Define request schema
class PredictionRequest(BaseModel):
    text: str

# Define response schema
class PredictionResponse(BaseModel):
    prediction: str
    final_score: float
    confidence: str

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the NLP Disease Prediction API. Send a POST request to /predict with JSON {'text': 'your symptoms'}"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_disease(req: PredictionRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    # Run the pipeline
    result = run_pipeline(req.text)
    
    # Extract the top ranked disease
    top_disease = result['ranked_diseases'][0]
    
    # Check if NO INDICATION
    if top_disease['confidence_level'] == "NO INDICATION":
        return PredictionResponse(
            prediction="NO INDICATION",
            final_score=top_disease['final_score'],
            confidence="NO INDICATION"
        )
        
    # Otherwise return the specific disease
    return PredictionResponse(
        prediction=top_disease['disease'],
        final_score=top_disease['final_score'],
        confidence=top_disease['confidence_level']
    )

if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)