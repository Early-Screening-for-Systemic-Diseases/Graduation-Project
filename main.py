import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as T
from model import HbNet

app = FastAPI(title="Anemia Detection API")

# ------------------ DEVICE ------------------
device = torch.device("cpu")

# ------------------ LOAD MODEL ------------------
model = HbNet()
model.load_state_dict(
    torch.load("best_hb_model.pth", map_location=device)
)
model.to(device)
model.eval()

# ------------------ TRANSFORM (MATCH TRAINING) ------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ UTILS ------------------
def anemia_from_hb(hb):
    return "anemic" if hb < 12.5 else "non_anemic"

# ------------------ ENDPOINT ------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        hb_pred = model(img_tensor).item()

    return {
        "hb_value": round(hb_pred, 2),
        "anemia_status": anemia_from_hb(hb_pred)
    }

# ------------------ HEALTH ------------------
@app.get("/health")
def health():
    return {"status": "healthy"}
