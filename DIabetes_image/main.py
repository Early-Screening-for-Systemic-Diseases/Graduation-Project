# main.py
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from DIabetes_image.model import DiabetesClassifier

app = FastAPI()

# -------- Load Model --------
device = torch.device("cpu")

# Load base MobileNetV2 (same as training)
base_model = models.mobilenet_v2(weights=None)

# Create classifier
model = DiabetesClassifier(base_model)

# Load trained weights
MODEL_PATH = "models/D_model.pt"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

# -------- TRANSFORM (MUST MATCH VALIDATION TRANSFORM) --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------- API Endpoint --------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()   # value between 0 and 1

    # Binary decision
    prediction = "non_diabetes" if prob >= 0.5 else "diabetes"

    return {
        "prediction": prediction,
        "probability_non_diabetes": prob
    }
