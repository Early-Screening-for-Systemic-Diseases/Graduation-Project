import io
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


app = FastAPI(title="Skin Cancer Classification API")

# ------------------ BUILD & LOAD MODEL ------------------
def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = True
    for layer in base_model.layers[:200]:
        layer.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs)


WEIGHTS_PATH = "best_model_weights.weights.h5"
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")

print("Building model and loading weights...")
model = build_model()
model.load_weights(WEIGHTS_PATH)
print("Model loaded successfully.")

# ------------------ CLASS MAPPING ------------------
INDEX_TO_CLASS = {0: "NV", 1: "MEL", 2: "BCC"}

# ------------------ IMAGE PREPROCESSING ------------------
def preprocess_image(image_array, target_size=224):
    """Preprocess image to match training pipeline."""
    image = tf.cast(image_array, tf.float32)

    # Resize shorter side to target_size
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    scale = tf.cast(target_size, tf.float32) / tf.minimum(h, w)
    new_h = tf.cast(tf.math.ceil(h * scale), tf.int32)
    new_w = tf.cast(tf.math.ceil(w * scale), tf.int32)
    image = tf.image.resize(image, [new_h, new_w], method="bilinear")

    # Center crop
    image = tf.image.resize_with_crop_or_pad(image, target_size, target_size)

    # Normalize to [0, 1] then apply EfficientNet preprocessing
    image = image / 255.0
    image = image * 255.0
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image

# ------------------ ENDPOINTS ------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict dermoscopic class from image."""
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        # Preprocess
        img_processed = preprocess_image(img_array)

        # Add batch dimension
        img_batch = tf.expand_dims(img_processed, axis=0)

        # Predict — model output is already softmax probabilities
        probabilities = model(img_batch, training=False).numpy()[0]
        predicted_class_idx = int(np.argmax(probabilities))
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