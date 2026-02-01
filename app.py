from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os

# -------------------- Flask App --------------------
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# -------------------- Device --------------------
device = torch.device("cpu")  # Railway = CPU only

# -------------------- Model Definition --------------------
class HbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc(feats)

# -------------------- Load Model --------------------
MODEL_PATH = "best_hb_model.pth"

model = HbNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# -------------------- Preprocessing (IDENTICAL LOGIC) --------------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def preprocess_image(file):
    img = Image.open(BytesIO(file.read())).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = transform(img)
    return img.unsqueeze(0)

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model": "ResNet18 Hb Regression",
        "task": "Anemia detection from eye images"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img_tensor = preprocess_image(request.files["file"]).to(device)

        with torch.no_grad():
            hb_pred = model(img_tensor).item()

        anemia = hb_pred < 12.5

        return jsonify({
            "status": "success",
            "predicted_hb": round(hb_pred, 2),
            "anemia": bool(anemia),
            "interpretation": "Anemic" if anemia else "Non-anemic",
            "threshold": 12.5,
            "note": "For research and educational purposes only"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
