# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

# Import your model class
from model import DiabetesClassifier
from torchvision.models import mobilenet_v2

app = FastAPI(title="Diabetes Detection API")

# -------------------------------
# Step 1 — Initialize model
# -------------------------------
# Use CPU-only
device = torch.device("cpu")

# Initialize base model
base_model = mobilenet_v2(weights=None)  # CPU-friendly
model = DiabetesClassifier(base_model)

# Load trained model
model.load_state_dict(torch.load("models/D_model.pt", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Step 2 — Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # make sure this matches your training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Step 3 — API endpoints
# -------------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0)  # add batch dimension
        image = image.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            pred = output.item()  # get scalar
            label = "non_diabetes" if pred >= 0.5 else "diabetes"
            confidence = float(pred) if pred >= 0.5 else float(1 - pred)

        return JSONResponse({
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
