# -----------------------------
# Disease Matcher Test in Notebook
# -----------------------------
import os
from pathlib import Path
import pandas as pd
from shared.disease_matcher_full import DiseaseMatcher

# -----------------------------
# Paths
# -----------------------------
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # parent of predictor.py
DATA_DIR = BASE_DIR / "data"

lexicon_path  = DATA_DIR / "lexicon.csv"
diabetes_path = DATA_DIR / "diabetes_symptoms.csv"
anemia_path   = DATA_DIR / "anemia_symptoms.csv"

# -----------------------------
# Initialize Matcher
# -----------------------------
matcher = DiseaseMatcher(
    lexicon_path=str(lexicon_path),
    diabetes_path=str(diabetes_path),
    anemia_path=str(anemia_path),
    semantic_model_name="all-MiniLM-L6-v2",
    semantic_threshold=0.55
)

def predict(text: str):
    results, lexicon_flag = matcher.match_disease(text)
    return {
        "text": text,
        "lexicon_matched": lexicon_flag,
        "results": results
    }
