# nlpv2/pipeline/biobert_branch.py
# Stage 3 — BioBERT Semantic Branch
#
# Uses: dmis-lab/biobert-base-cased-v1.2 via sentence-transformers
# Model is downloaded on first use (~1.3 GB) and cached by HuggingFace.
# Model is loaded ONCE at module startup — never inside run_pipeline().

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .lexicon_branch import disease_to_symptoms

# ── Constants ────────────────────────────────────────────────────────────────
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
BIOBERT_THRESHOLD  = 0.55

# ── Load BioBERT once at module startup ──────────────────────────────────────
print(f"[biobert_branch] Loading BioBERT model: {BIOBERT_MODEL_NAME} ...")
_biobert = SentenceTransformer(BIOBERT_MODEL_NAME)
print("[biobert_branch] BioBERT model loaded.")


def build_disease_embeddings(d_to_s: dict = None) -> dict:
    """
    Pre-compute a single embedding vector per disease by encoding
    the concatenation of all its canonical symptoms.

    Args:
        d_to_s: disease_to_symptoms dict (uses module-level default).

    Returns:
        { 'Diabetes': np.ndarray(shape=(768,)), 'Skin Cancer': np.ndarray(...) }
    """
    d_to_s = d_to_s or disease_to_symptoms
    embeddings = {}
    for disease, syms in d_to_s.items():
        combined = ", ".join(syms)
        embeddings[disease] = _biobert.encode(combined)
    return embeddings


# Pre-compute disease embeddings once at startup
print("[biobert_branch] Pre-computing disease embeddings...")
_disease_embeddings = build_disease_embeddings()
print("[biobert_branch] Disease embeddings ready.")


def biobert_branch(clean_sentence: str, disease_embeddings: dict = None) -> dict:
    """
    Stage 3 — BioBERT Branch.

    Encodes the patient sentence and computes cosine similarity against
    each disease's pre-computed embedding vector.

    Args:
        clean_sentence: The preprocessed patient sentence (from Stage 1).
        disease_embeddings: Pre-computed disease embedding dict (uses module-level default).

    Returns:
        {
          'Diabetes': {
            'similarity': float,   # raw cosine similarity [−1, 1]
            'matched': bool        # True if similarity >= BIOBERT_THRESHOLD
          },
          'Skin Cancer': { ... }
        }
    """
    disease_embeddings = disease_embeddings or _disease_embeddings

    # Encode the patient sentence
    sentence_vector = _biobert.encode(clean_sentence)

    results = {}
    for disease, disease_vector in disease_embeddings.items():
        sim = float(
            cosine_similarity(
                sentence_vector.reshape(1, -1),
                disease_vector.reshape(1, -1)
            )[0][0]
        )
        results[disease] = {
            "similarity": sim,
            "matched": sim >= BIOBERT_THRESHOLD,
        }

    return results
