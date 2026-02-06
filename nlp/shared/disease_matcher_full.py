# disease_matcher_full.py
import os
import pandas as pd
from pathlib import Path
from nlp.shared.lexicon_loader import load_lexicon
from nlp.shared.symptom_matcher import match_symptoms
from nlp.shared.preprocessing import clean_text
from nlp.shared.lemmatizer import lemmatize_text

# Semantic Matching (SBERT or Zero-Shot)
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 1️⃣ Helper Functions
# -----------------------------
def normalize_symptom(s: str) -> str:
    return s.strip().lower()

def load_disease_symptoms(diabetes_df, anemia_df):
    disease_symptoms = {}
    for df in [diabetes_df, anemia_df]:
        for _, row in df.iterrows():
            disease = row["disease"].strip().lower()
            symptoms = {normalize_symptom(s) for s in row[1:] if pd.notna(s)}
            disease_symptoms[disease] = symptoms
    return disease_symptoms

# -----------------------------
# 2️⃣ Semantic Matcher Setup
# -----------------------------
class SemanticMatcher:
    def __init__(self, disease_symptoms, model_name="all-MiniLM-L6-v2", threshold=0.55):
        self.model = SentenceTransformer(model_name)
        # Flatten all unique symptoms across diseases
        self.symptoms = sorted({s for sset in disease_symptoms.values() for s in sset})
        self.symptom_embeddings = self.model.encode(self.symptoms, convert_to_tensor=True)
        self.threshold = threshold

    def match(self, text):
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        cos_scores = util.cos_sim(text_embedding, self.symptom_embeddings)[0]
        results = []
        for idx, score in enumerate(cos_scores):
            if score >= self.threshold:
                results.append((self.symptoms[idx], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# -----------------------------
# 3️⃣ Full Disease Matcher
# -----------------------------
class DiseaseMatcher:
    def __init__(self, lexicon_path, diabetes_path, anemia_path, semantic_model_name="all-MiniLM-L6-v2", semantic_threshold=0.55):
        # Paths
        self.lexicon_path = lexicon_path
        self.diabetes_path = diabetes_path
        self.anemia_path = anemia_path

        # Load lexicon
        self.lexicon = load_lexicon(str(self.lexicon_path))

        # Load disease CSVs
        diabetes_df = pd.read_csv(self.diabetes_path)
        anemia_df   = pd.read_csv(self.anemia_path)
        self.disease_symptoms = load_disease_symptoms(diabetes_df, anemia_df)

        # Semantic matcher
        self.semantic_matcher = SemanticMatcher(self.disease_symptoms, model_name=semantic_model_name, threshold=semantic_threshold)

    def match_disease(self, text):
        # 1️⃣ Lexicon matching
        detected_symptoms = match_symptoms(text, self.lexicon)
        detected_symptoms = [normalize_symptom(s) for s in detected_symptoms]
        lexicon_flag = len(detected_symptoms) > 0

        # If lexicon found nothing → fallback
        if not lexicon_flag:
            semantic_results = self.semantic_matcher.match(text)
            detected_symptoms = [normalize_symptom(s[0]) for s in semantic_results]

        # Map symptoms to diseases
        results = []
        detected_set = set(detected_symptoms)
        for disease, disease_set in self.disease_symptoms.items():
            matched = detected_set & disease_set
            if matched:
                percentage = len(matched) / len(disease_set) * 100
                results.append({
                    "disease": disease,
                    "matched_symptoms": sorted(matched),
                    "percentage": round(percentage, 2)
                })
        results.sort(key=lambda x: x["percentage"], reverse=True)
        return results, lexicon_flag
