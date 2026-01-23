# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from model import DiabetesClassifier  # your model class
import torchvision.models as models

# ------------------------------
# Initialize FastAPI app
# ------------------------------
app = FastAPI(
    title="Diabetes Detection API",
    description="Upload an image of tongue to predict Diabetes presence",
    version="1.0"
)

# ------------------------------
# Load model on CPU
# ------------------------------
# Initialize base model (same as used during training)
base_model = models.mobilenet_v2(weights=None)  # CPU only
model = DiabetesClassifier(base_model)

# Load saved state dict
model_path = "models/D_model.pt"  # ensure the file exists in repo
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.to(torch.device("cpu"))
model.eval()

# ------------------------------
# Image preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Apply transforms
        image = transform(image).unsqueeze(0)  # add batch dimension

        # Prediction
        with torch.no_grad():
            output = model(image)
            prediction = output.item()
            label = "Non-Diabetes" if prediction >= 0.5 else "Diabetes"

        return JSONResponse(content={
            "prediction": label,
            "score": round(prediction, 4)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
