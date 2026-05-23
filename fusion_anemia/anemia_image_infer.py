# anemia_image_infer.py

import torch
import math
from PIL import Image
from torchvision import transforms


# ============================================
# STEP 1: DEFINE YOUR MODEL CLASS
# Copy your HbNet class EXACTLY from training
# ============================================

class HbNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Example structure — REPLACE with your exact structure if different
        self.backbone = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(224*224*3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================
# STEP 2: PUT YOUR SAVED MODEL PATH HERE
# ============================================

MODEL_PATH = r"CD:\01Grad Project\Graduation-Project\Anemia_Images\best_hb_model.pth"
#            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# REPLACE THIS with your actual path
#
# Example:
# MODEL_PATH = r"C:\Users\Abdelrahman\Desktop\models\best_hb_model.pth"
#
# OR if inside project folder:
# MODEL_PATH = "models/best_hb_model.pth"


# ============================================
# STEP 3: LOAD MODEL
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HbNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# ============================================
# STEP 4: IMAGE PREPROCESSING
# MUST match training preprocessing
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================
# STEP 5: CONVERT Hb → anemia probability
# ============================================

def hb_to_probability(hb_value, gender):
    
    if gender.lower() == "male":
        threshold = 13.0
    else:
        threshold = 12.0
    
    temp = 1.0
    
    probability = 1 / (1 + math.exp((hb_value - threshold) / temp))
    
    return probability


# ============================================
# STEP 6: MAIN INFERENCE FUNCTION
# ============================================

def predict_anemia_from_image(image_path, gender):
    
    image = Image.open(image_path).convert("RGB")
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        hb_value = model(tensor).item()
    
    probability = hb_to_probability(hb_value, gender)
    
    return probability



