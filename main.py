import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from model import DiabetesClassifier
import io

app = FastAPI()

device = torch.device("cpu")
MODEL_PATH = "models/D_model.pt"

# Load MobileNetV2 base model (NO pretrained)
base_model = models.mobilenet_v2(pretrained=False)

# Create classifier
model = DiabetesClassifier(base_model)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image preprocessing (MUST match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"status": "Diabetes Detection API Running"}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Read & Process
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        # 2. Predict (BINARY LOGIC)
        with torch.no_grad():
            outputs = model(tensor)
            
            # Use Sigmoid for 1-output models
            # Result is a probability between 0.0 and 1.0
            prob = torch.sigmoid(outputs).item()

        # 3. Determine Class (Threshold usually 0.5)
        # If prob > 0.5, it's the second class (Index 1)
        # If prob < 0.5, it's the first class (Index 0)
        
        if prob > 0.5:
            predicted_index = 1
            confidence = prob
        else:
            predicted_index = 0
            confidence = 1 - prob # Invert it (e.g., if prob is 0.1, confidence is 90% for class 0)

        class_names = ["diabetes", "non_diabetes"]
        
        return {
            "prediction": class_names[predicted_index],
            "confidence_score": f"{confidence * 100:.2f}%",
            "raw_output": prob,
            "logic_used": "Binary Sigmoid (< 0.5 = diabetes)"
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}