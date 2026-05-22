# nlpv2/pipeline/spacy_correction.py
# Stage 6 — SpaCy Post-Correction
#
# CRITICAL RULES (from spec):
#   - Apply corrections ONLY to Lexicon and BioBERT matched symptoms.
#   - NEVER touch BART scores — BART is a black-box NLI model.
#   - Negation subtracts the IDF-weighted contribution of the negated symptom
#     from the fused score.
#   - Intensity multiplies the final score (HIGH=1.3x, MILD=0.8x, NONE=1.0x).

import spacy
import subprocess
import sys

# ── Intensity word sets ──────────────────────────────────────────────────────
HIGH_WORDS = {
    "extreme", "severe", "intense", "chronic", "constant",
    "terrible", "horrible", "unbearable", "persistent", "very",
    "always", "continuously", "excruciating",
}

MILD_WORDS = {
    "mild", "slight", "occasional", "minor", "little",
    "bit", "sometimes", "rarely", "barely", "somewhat",
    "a bit", "not too bad",
}

# ── Load SpaCy + negspacy once at module startup ─────────────────────────────
try:
    _nlp_correction = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )
    _nlp_correction = spacy.load("en_core_web_sm")

try:
    from negspacy.negation import Negex
    if "negex" not in _nlp_correction.pipe_names:
        _nlp_correction.add_pipe("negex")
    _has_negspacy = True
    print("[spacy_correction] negspacy loaded successfully.")
except ImportError:
    _has_negspacy = False
    print("[spacy_correction] Warning: negspacy not installed. "
          "Falling back to simple negation word detection.")


# ── Negation Detection ───────────────────────────────────────────────────────

def _simple_negation_fallback(clean_sentence: str, all_symptoms: list) -> set:
    """
    Fallback negation detection when negspacy is unavailable.
    Looks for negation trigger words (no, not, never, without, don't, etc.)
    within a 3-token window before a symptom word.
    """
    NEG_TRIGGERS = {"no", "not", "never", "without", "deny", "denies",
                    "denied", "don't", "doesn't", "do not", "does not"}
    doc = _nlp_correction(clean_sentence)
    tokens = [t.lemma_.lower() for t in doc]
    negated = set()
    for sym in all_symptoms:
        sym_tokens = sym.lower().replace("_", " ").split()
        for i, tok in enumerate(tokens):
            if tok in sym_tokens:
                window = tokens[max(0, i - 4): i]
                if any(neg in " ".join(window) for neg in NEG_TRIGGERS):
                    negated.add(sym)
                    break
    return negated


def detect_negated_symptoms(clean_sentence: str, all_symptoms: list) -> set:
    """
    Detect which canonical symptoms in the sentence are negated.

    Uses negspacy (dependency-tree-based) when available,
    falls back to simple window-based detection otherwise.

    Args:
        clean_sentence: Preprocessed patient sentence.
        all_symptoms:   List of all canonical symptom name strings.

    Returns:
        Set of negated canonical symptom names.
    """
    if not _has_negspacy:
        return _simple_negation_fallback(clean_sentence, all_symptoms)

    doc = _nlp_correction(clean_sentence)
    negated = set()
    for token in doc:
        if token._.negex:
            tok_lemma = token.lemma_.lower()
            for sym in all_symptoms:
                # Match if any word of the symptom key appears in the token
                sym_parts = sym.lower().replace("_", " ").split()
                if tok_lemma in sym_parts:
                    negated.add(sym)
    return negated


# ── Intensity Detection ──────────────────────────────────────────────────────

def detect_intensity(clean_sentence: str) -> str:
    """
    Detect the overall intensity/severity described in the sentence.

    Returns:
        'HIGH'  — if high-intensity words are present
        'MILD'  — if mild-intensity words are present (and no HIGH words)
        'NONE'  — if no intensity modifier is found
    """
    doc = _nlp_correction(clean_sentence)
    lemmas = {t.lemma_.lower() for t in doc}

    if lemmas & HIGH_WORDS:
        return "HIGH"
    if lemmas & MILD_WORDS:
        return "MILD"
    return "NONE"


# ── Stage 6 Main Function ────────────────────────────────────────────────────

def spacy_post_correct(
    fused:        dict,
    lex_results:  dict,
    clean_sentence: str,
    idf_weights:  dict,
) -> dict:
    """
    Stage 6 — SpaCy Post-Correction.

    Applies two corrections to the fused scores:
      1. Negation penalty: subtracts the IDF-weighted lexicon contribution
         of any negated symptom from the fused score.
      2. Intensity multiplier: scales the final score up (HIGH) or down (MILD).

    CRITICAL: Corrections are computed from Lexicon matched symptoms only.
              BART scores are NEVER modified here.

    Args:
        fused:          Output of entropy_fusion() — contains 'fused_score' and 'weights'.
        lex_results:    Output of lexicon_branch() — contains 'matched_symptoms'.
        clean_sentence: Preprocessed patient sentence.
        idf_weights:    {symptom: idf_weight} from compute_idf_weights().

    Returns:
        Corrected fused dict with additional fields:
        {
          'Diabetes': {
            ...all fused fields...,
            'final_score': float,
            'negated_symptoms': [str, ...],
            'intensity': str,
            'multiplier': float
          },
          ...
        }
    """
    all_symptoms = list(idf_weights.keys())
    negated   = detect_negated_symptoms(clean_sentence, all_symptoms)
    intensity = detect_intensity(clean_sentence)

    intensity_multipliers = {"HIGH": 1.3, "MILD": 0.8, "NONE": 1.0}
    mult = intensity_multipliers[intensity]

    corrected = {}
    for disease, res in fused.items():
        score = res["fused_score"]

        # Lexicon matched symptoms for this disease
        lex_syms = [s["canonical"] for s in lex_results[disease]["matched_symptoms"]]
        lex_w_total = sum(idf_weights.get(s, 0.0) for s in lex_syms)

        # Negation penalty — subtract the IDF-weighted contribution of each
        # negated symptom that was matched by the Lexicon branch
        for sym in negated:
            if sym in lex_syms and lex_w_total > 0:
                # Contribution = lexicon branch weight × (sym_idf / total_lex_idf)
                contrib = res["weights"]["lexicon"] * (
                    idf_weights.get(sym, 0.0) / lex_w_total
                )
                score = max(0.0, score - contrib)

        # Intensity multiplier — clamp to [0, 1]
        score = min(1.0, score * mult)

        corrected[disease] = {
            **res,
            "final_score":       round(score, 6),
            "negated_symptoms":  sorted(list(negated)),
            "intensity":         intensity,
            "multiplier":        mult,
        }

    return corrected
