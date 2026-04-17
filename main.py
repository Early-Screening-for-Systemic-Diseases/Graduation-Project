import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = FastAPI(title="Anemia Detection API")

# ------------------ LOAD MODEL ------------------
model = keras.models.load_model("final_model.keras")

# ------------------ CLASS MAPPING ------------------
INDEX_TO_CLASS = {0: "benign", 1: "malignant", 2: "melanoma"}

# ------------------ IMAGE PREPROCESSING ------------------
def preprocess_image(image_array, target_size=224):
    """Preprocess image to match training pipeline."""
    # Resize with crop-or-pad to target size
    h, w = image_array.shape[:2]
    scale = target_size / min(h, w)
    new_h = int(np.ceil(h * scale))
    new_w = int(np.ceil(w * scale))
    image = tf.image.resize(image_array, [new_h, new_w], method="bilinear")
    image = tf.image.resize_with_crop_or_pad(image, target_size, target_size)
    
    # Convert to [0, 1]
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)
    if tf.reduce_max(image) > 1.0:
        image = image / 255.0
    
    # Scale to [0, 255] for EfficientNet preprocessing
    image = image * 255.0
    
    # Apply EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    
    return image

# ------------------ ENDPOINTS ------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict dermoscopic class from image."""
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Preprocess
        img_processed = preprocess_image(img_array)
        
        # Add batch dimension
        img_batch = tf.expand_dims(img_processed, axis=0)
        
        # Predict
        logits = model(img_batch, training=False)
        probabilities = tf.nn.softmax(logits).numpy()[0]
        predicted_class_idx = int(tf.argmax(logits[0]))
        predicted_class = INDEX_TO_CLASS[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                INDEX_TO_CLASS[i]: round(float(prob), 4) 
                for i, prob in enumerate(probabilities)
            }
        }
    except Exception as e:
        return {"error": str(e)}

# ------------------ HEALTH ------------------
@app.get("/health")
def health():
    return {"status": "healthy"}
