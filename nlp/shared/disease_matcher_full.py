# disease_matcher_full.py
import os
import pandas as pd
from pathlib import Path
from nlp.shared.lexicon_loader import load_lexicon
from nlp.shared.symptom_matcher import match_symptoms
from nlp.shared.preprocessing import clean_text
from nlp.shared.lemmatizer import lemmatize_text

# Semantic Matching (SBERT or Zero-Shot)
from nlp.shared.semantic_embedder_onnx import ONNXEmbedder
import numpy as np

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
# -----------------------------
# 2️⃣ Semantic Matcher Setup (ONNX)
# -----------------------------


class SemanticMatcher:
    def __init__(self, disease_symptoms, model_name=None, onnx_model_dir="onnx_model", threshold=0.55, **kwargs):
        # Accept `model_name` for backwards-compatibility with other callers
        # If a model_name string is provided and there's a matching folder, prefer it
        if model_name:
            candidate = Path(model_name)
            if candidate.exists():
                onnx_dir = str(candidate)
            else:
                # keep using provided onnx_model_dir
                onnx_dir = onnx_model_dir
        else:
            onnx_dir = onnx_model_dir

        self.embedder = ONNXEmbedder(onnx_dir)
        # Keep disease order stable
        self.diseases = sorted(disease_symptoms.keys())
        # Symptoms grouped per disease (list of lists)
        self.symptoms_by_disease = [sorted(list(disease_symptoms[d])) for d in self.diseases]
        # Compute embeddings per symptom and a single vector per disease by averaging
        disease_vectors = []
        # store per-disease symptom embeddings for later per-symptom matching
        self.symptom_embeddings_by_disease = []
        for symptom_list in self.symptoms_by_disease:
            if len(symptom_list) == 0:
                disease_vectors.append(None)
                self.symptom_embeddings_by_disease.append(None)
                continue
            emb = self.embedder.encode(symptom_list)
            # If tokenizer/encoder returns token-level embeddings (3D), average tokens
            if emb.ndim == 3:
                emb = emb.mean(axis=1)
            # emb now should be (n_symptoms, dim)
            self.symptom_embeddings_by_disease.append(emb)
            # Now average across symptoms to get disease vector
            disease_vectors.append(emb.mean(axis=0))

        # Replace any None entries with zeros of appropriate dimension
        if any(v is None for v in disease_vectors):
            # infer dim from first non-None
            non_none = next((v for v in disease_vectors if v is not None), None)
            if non_none is None:
                # no embeddings at all; create empty array
                self.disease_vectors = np.zeros((len(disease_vectors), 1))
            else:
                dim = non_none.shape[-1]
                disease_vectors = [np.zeros(dim) if v is None else v for v in disease_vectors]
                self.disease_vectors = np.vstack(disease_vectors)
        else:
            self.disease_vectors = np.vstack(disease_vectors)

        self.threshold = threshold

    def match_symptoms_for_text(self, text):
        """Return per-disease list of (symptom, score) pairs using cosine similarity

        Returns: dict disease -> list of (symptom, score)
        """
        # Encode text
        text_embedding = self.embedder.encode([text])
        if text_embedding.ndim == 3:
            text_embedding = text_embedding.mean(axis=1)
        text_vector = text_embedding.mean(axis=0)

        # Norm of text vector
        tnorm = np.linalg.norm(text_vector)
        if tnorm == 0:
            tnorm = 1e-8

        results = {}
        for i, disease in enumerate(self.diseases):
            sym_emb = self.symptom_embeddings_by_disease[i]
            if sym_emb is None:
                results[disease] = []
                continue
            # compute cosine between each symptom embedding and text_vector
            denom = (np.linalg.norm(sym_emb, axis=1) * tnorm)
            denom[denom == 0] = 1e-8
            scores = np.dot(sym_emb, text_vector) / denom
            # pair symptom text with score
            paired = [(self.symptoms_by_disease[i][j], float(scores[j])) for j in range(len(scores))]
            # sort by score desc
            paired.sort(key=lambda x: x[1], reverse=True)
            results[disease] = paired

        return results

    def match(self, text):
        # Encode the text
        text_embedding = self.embedder.encode([text])
        if text_embedding.ndim == 3:
            text_embedding = text_embedding.mean(axis=1)
        # text_embedding may be (1, dim) or (n, dim); average to single vector
        text_vector = text_embedding.mean(axis=0)

        # Cosine similarity between each disease vector and the text vector
        denom = (np.linalg.norm(self.disease_vectors, axis=1) * np.linalg.norm(text_vector))
        # avoid div-by-zero
        denom[denom == 0] = 1e-8
        cos_scores = np.dot(self.disease_vectors, text_vector) / denom

        return {disease: float(score) for disease, score in zip(self.diseases, cos_scores)}


    
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

        # Semantic matcher (uses local ONNX directory by default)
        # Forward semantic_model_name (if provided) for compatibility
        self.semantic_matcher = SemanticMatcher(
            self.disease_symptoms,
            model_name=semantic_model_name,
            onnx_model_dir="onnx_model",
            threshold=semantic_threshold,
        )

    def match_disease(self, text):
        # 1️⃣ Lexicon matching
        detected_symptoms = match_symptoms(text, self.lexicon)
        detected_symptoms = [normalize_symptom(s) for s in detected_symptoms]
        lexicon_flag = len(detected_symptoms) > 0

        # If lexicon found nothing → fallback using semantic matcher
        if not lexicon_flag:
            # get per-symptom semantic scores
            per_symptom = self.semantic_matcher.match_symptoms_for_text(text)

            # Find symptoms above threshold and compute matched counts per disease
            any_matched = False
            results = []
            for disease, pairs in per_symptom.items():
                # pairs is list of (symptom, score)
                matched_symptoms = [s for s, sc in pairs if sc >= self.semantic_matcher.threshold]
                if matched_symptoms:
                    any_matched = True
                    disease_set = self.disease_symptoms.get(disease, set())
                    # compute percentage as matched canonical symptoms / total symptoms
                    total = len(disease_set) if disease_set else 1
                    percentage = len(set(matched_symptoms) & disease_set) / total * 100
                    results.append({
                        "disease": disease,
                        "matched_symptoms": sorted(list(set(matched_symptoms) & disease_set)),
                        "percentage": round(percentage, 2)
                    })

            # If we found any symptom-level semantic matches, return them sorted
            if any_matched:
                results.sort(key=lambda x: x["percentage"], reverse=True)
                return results, False

            # No symptom passed threshold — choose top disease and include its top symptom (best-effort)
            semantic_scores = self.semantic_matcher.match(text)
            if semantic_scores:
                top = max(semantic_scores.items(), key=lambda x: x[1])[0]
                # top disease's top symptom (highest score)
                top_pairs = per_symptom.get(top, [])
                if top_pairs:
                    top_symptom = top_pairs[0][0]
                    disease_set = self.disease_symptoms.get(top, set())
                    total = len(disease_set) if disease_set else 1
                    percentage = (1 / total) * 100
                    return [{
                        "disease": top,
                        "matched_symptoms": [top_symptom],
                        "percentage": round(percentage, 2)
                    }], False
            # fallback: no semantic info
            return [], False

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
