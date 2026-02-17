from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from app.model_loader import model
from app.preprocessing import TonguePreprocessor


app = FastAPI(title="Diabetes Image Prediction API")

processor = TonguePreprocessor()

@app.get("/")
def root():
    return {"message": "Diabetes Image Model is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = processor.process_image(img_rgb)
    img_array = np.expand_dims(processed, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        label = "Diabetic"
        confidence = float((1 - prediction) * 100)
    else:
        label = "Non-Diabetic"
        confidence = float(prediction * 100)

    return {
        "prediction": label,
        "confidence_percentage": round(confidence, 2)
    }
