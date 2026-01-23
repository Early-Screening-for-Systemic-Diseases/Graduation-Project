from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

app = FastAPI()

# Path to model
MODEL_PATH = os.path.join("models", "diabetes_model.pt")

# Load model on CPU
device = "cpu"
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict-image")
async def predict_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output).item()

    return {"prediction": int(prediction)}
