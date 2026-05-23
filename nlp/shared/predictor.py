# -----------------------------
# Disease Matcher Test in Notebook
# -----------------------------
import os
from pathlib import Path
import pandas as pd
from nlp.shared.disease_matcher_full import DiseaseMatcher

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

    # Ensure we return the detailed list of results under `results`
    detailed_results = results if isinstance(results, list) else []

    # Build a compatibility mapping: disease -> { percentage, matched_symptoms }
    results_map = {}
    if isinstance(results, dict):
        # older callers may receive a dict disease->score
        for disease, score in results.items():
            try:
                pct = float(score)
            except Exception:
                pct = 0.0
            results_map[disease] = {"percentage": pct, "matched_symptoms": []}
    elif isinstance(detailed_results, list):
        for r in detailed_results:
            if not isinstance(r, dict):
                continue
            name = r.get("disease") or "unknown"
            pct = r.get("percentage") if r.get("percentage") is not None else r.get("score")
            try:
                pct = float(pct) if pct is not None else 0.0
            except Exception:
                pct = 0.0
            matched = r.get("matched_symptoms") or []
            results_map[name] = {"percentage": pct, "matched_symptoms": list(matched)}
    else:
        # fallback
        results_map = {}

    return {
        "text": text,
        "lexicon_matched": lexicon_flag,
        "results": detailed_results,
        "results_map": results_map
    }
