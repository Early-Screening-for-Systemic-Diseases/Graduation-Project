from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import os

app = FastAPI()

# -----------------------------
# Load model at startup (SAFE FOR RAILWAY)
# -----------------------------
model = None

@app.on_event("startup")
def load_my_model():
    global model
    MODEL_PATH = "best_diabetes_model"
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")

# -----------------------------
# Class Mapping
# -----------------------------
CLASS_MAP = {
    1: "Non-Diabetic",
    0: "Diabetic"
}

# -----------------------------
# Preprocessing Class
# -----------------------------
class TonguePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def remove_shadows(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = self.remove_shadows(img)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 40, 50])
        upper1 = np.array([20, 255, 255])
        lower2 = np.array([160, 40, 50])
        upper2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            x = max(0, x-10)
            y = max(0, y-10)
            img = img[y:y+h+20, x:x+w+20]

        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype('float32'))

        return img

processor = TonguePreprocessor()

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)

    processed = processor.process_image(image)
    input_tensor = np.expand_dims(processed, axis=0)

    prediction = model.predict(input_tensor)[0][0]

    if prediction > 0.5:
        label = CLASS_MAP[1]
        confidence = float(prediction)
    else:
        label = CLASS_MAP[0]
        confidence = float(1 - prediction)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }

@app.get("/")
def home():
    return {"message": "Diabetes Detection API is running"}