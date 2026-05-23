# anemia_questionnaire_infer.py

import joblib
import numpy as np


# ============================================
# PUT YOUR SAVED MODEL PATH HERE
# ============================================

MODEL_PATH = r"C:\Users\YourName\project\models\anemia_questionnaire.pkl"
# REPLACE THIS PATH


# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(MODEL_PATH)


# ============================================
# FEATURE ORDER
# MUST match training exactly
# ============================================

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Fatigue",
    "Pale_skin",
    "Shortness_of_breath",
    "Dizziness"
]


# ============================================
# INFERENCE FUNCTION
# ============================================

def predict_anemia_from_questionnaire(features_dict):
    
    x = np.array([[features_dict[f] for f in FEATURE_ORDER]])
    
    probability = model.predict_proba(x)[0, 1]
    
    return probability


# Example
if __name__ == "__main__":
    
    sample = {
        "Age": 22,
        "Gender": 1,
        "Fatigue": 1,
        "Pale_skin": 0,
        "Shortness_of_breath": 1,
        "Dizziness": 0
    }
    
    prob = predict_anemia_from_questionnaire(sample)
    
    print("Questionnaire probability:", prob)
