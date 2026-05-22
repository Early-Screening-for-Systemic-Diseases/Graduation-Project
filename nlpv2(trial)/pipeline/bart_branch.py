# nlpv2/pipeline/bart_branch.py
# Stage 4 — BART NLI Zero-Shot Branch
#
# Uses: facebook/bart-large-mnli via HuggingFace transformers pipeline.
# Model is downloaded on first use (~1.6 GB) and cached by HuggingFace.
# Model is loaded ONCE at module startup — never inside run_pipeline().
#
# CRITICAL RULES (from spec):
#   - Use multi_label=False (scores are already softmax probabilities).
#   - Apply temperature scaling ONLY to BART scores, never to other branches.
#   - Do NOT change TEMPERATURE (1.5) without documenting the reason.

from transformers import pipeline as hf_pipeline

# ── Constants ────────────────────────────────────────────────────────────────
BART_MODEL_NAME = "facebook/bart-large-mnli"

# Temperature > 1.0 softens the distribution (reduces overconfidence).
# Value of 1.5 was chosen by the spec author to calibrate BART's tendency
# to produce very high confidence scores on short medical texts.
TEMPERATURE = 1.5

# ── Load BART NLI once at module startup ─────────────────────────────────────
print(f"[bart_branch] Loading BART NLI model: {BART_MODEL_NAME} ...")
_nli = hf_pipeline("zero-shot-classification", model=BART_MODEL_NAME)
print("[bart_branch] BART NLI model loaded.")


def bart_branch(clean_sentence: str, diseases: list) -> dict:
    """
    Stage 4 — BART NLI Branch.

    Runs zero-shot classification to determine the probability that the
    patient sentence entails each disease hypothesis.

    Temperature scaling is applied to the raw BART scores to soften
    overconfident predictions:
        calibrated_score = raw_score ^ (1 / TEMPERATURE)

    Args:
        clean_sentence: The preprocessed patient sentence (from Stage 1).
        diseases: List of disease names, e.g. ['Diabetes', 'Skin Cancer'].

    Returns:
        {
          'Diabetes': {
            'raw_score': float,        # original softmax probability from BART
            'calibrated_score': float  # temperature-scaled score
          },
          'Skin Cancer': { ... }
        }
    """
    # Build natural-language hypotheses from disease names
    labels = [f"This person shows signs of {d}" for d in diseases]

    # multi_label=False → scores sum to 1 (softmax distribution)
    out = _nli(clean_sentence, candidate_labels=labels, multi_label=False)

    results = {}
    for label, score in zip(out["labels"], out["scores"]):
        # Strip the hypothesis wrapper to recover the disease name
        disease = label.replace("This person shows signs of ", "")
        raw = float(score)
        calibrated = float(raw ** (1.0 / TEMPERATURE))
        results[disease] = {
            "raw_score": raw,
            "calibrated_score": calibrated,
        }

    return results
