import tensorflow as tf

MODEL_PATH = "models/best_diabetes_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)
