from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load once at module startup
biobert = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')

BIOBERT_THRESHOLD = 0.55

def build_disease_embeddings(disease_to_symptoms):
    disease_level = {}
    symptom_level = {}
    for d, syms in disease_to_symptoms.items():
        disease_level[d] = biobert.encode(', '.join(syms))
        symptom_level[d] = {
            sym: biobert.encode(sym) for sym in syms
        }
    return disease_level, symptom_level

def biobert_branch(clean_sentence, disease_embeddings, symptom_embeddings):
    sv = biobert.encode(clean_sentence)
    results = {}
    for disease, dv in disease_embeddings.items():
        sim = float(cosine_similarity(sv.reshape(1,-1), dv.reshape(1,-1))[0][0])
        # per-symptom breakdown
        matched_syms = []
        for sym, sym_vec in symptom_embeddings[disease].items():
            sym_sim = float(cosine_similarity(sv.reshape(1,-1), sym_vec.reshape(1,-1))[0][0])
            if sym_sim >= BIOBERT_THRESHOLD:
                matched_syms.append({'canonical': sym, 'similarity': sym_sim})
        results[disease] = {
            'similarity': sim,
            'matched': sim >= BIOBERT_THRESHOLD,
            'matched_symptoms': matched_syms
        }
    return results