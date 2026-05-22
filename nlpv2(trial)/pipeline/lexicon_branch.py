# nlpv2/pipeline/lexicon_branch.py
# Stage 2 — Lexicon Branch with IDF-weighted scoring
#
# Reads two sources (both copied from nlp/data/ — no changes to originals):
#   1. lexicon.csv      → canonical symptom → patient expressions
#   2. *_symptoms.csv   → disease → list of canonical symptoms
#
# Builds at startup:
#   symptom_to_diseases : {symptom_name: [Disease, ...]}
#   disease_to_symptoms : {Disease: [symptom_name, ...]}
#   lexicon             : {frozenset(lemma_tokens): canonical_symptom}

import re
import math
import pandas as pd
import spacy
import subprocess
import sys
from pathlib import Path

# ── Load SpaCy once (same model as Stage 1) ─────────────────────────────────
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    _nlp = spacy.load("en_core_web_sm")

# ── Paths (relative to this file's location) ────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_LEXICON_PATH       = _DATA_DIR / "lexicon.csv"
_DISEASE_PATHS      = [
    _DATA_DIR / "diabetes_symptoms.csv",
    _DATA_DIR / "skincancer_symptoms.csv",
]


# ── Helper: lemmatize a short expression string ──────────────────────────────
def _lemmatize_expression(expr: str) -> frozenset:
    """
    Lemmatize a patient expression and return a frozenset of tokens.
    e.g. "peeing a lot" → frozenset({'pee', 'a', 'lot'})
    Only include non-punct, non-space tokens.
    """
    doc = _nlp(expr.lower().strip())
    tokens = {t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space and t.lemma_.strip()}
    return frozenset(tokens) if tokens else None


def _normalize_symptom_name(name: str) -> str:
    """Normalize a symptom name to lowercase with underscores."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


# ── Build data structures at module startup ──────────────────────────────────

def _load_disease_to_symptoms(disease_paths: list) -> dict:
    """
    Read wide-format disease CSVs (disease, symptom, symptom, ...).
    Returns: { 'Diabetes': ['fatigue', 'polyuria', ...], 'Skin Cancer': [...] }
    """
    d2s = {}
    for path in disease_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[lexicon_branch] Warning: could not read {path}: {e}")
            continue

        for _, row in df.iterrows():
            disease_raw = str(row.iloc[0]).strip()
            # Normalize disease name to title case
            disease = disease_raw.title()
            symptoms = []
            for val in row.iloc[1:]:
                if pd.notna(val) and str(val).strip():
                    symptoms.append(_normalize_symptom_name(str(val)))
            if symptoms:
                d2s[disease] = symptoms
    return d2s


def _load_lexicon(lexicon_path: Path) -> dict:
    """
    Read the wide-format lexicon CSV:
        Canonical Symptom | Patient Expression 1 | Patient Expression 2 | ...

    Returns: { canonical_symptom_normalized: [expression_str, ...] }
    """
    try:
        df = pd.read_csv(lexicon_path, encoding="cp1252")
    except Exception:
        df = pd.read_csv(lexicon_path, encoding="utf-8", errors="replace")

    # Strip BOM and whitespace from column names
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    df = df.fillna("")

    result = {}
    canonical_col = df.columns[0]  # "Canonical Symptom"
    for _, row in df.iterrows():
        canonical = _normalize_symptom_name(str(row[canonical_col]))
        if not canonical:
            continue
        expressions = [str(v).strip() for v in row.iloc[1:] if str(v).strip()]
        result[canonical] = expressions
    return result


def _build_structures(disease_paths: list, lexicon_path: Path):
    """
    Build and return the three core data structures needed by the pipeline.
    """
    disease_to_symptoms = _load_disease_to_symptoms(disease_paths)

    # symptom_to_diseases: invert disease_to_symptoms
    symptom_to_diseases: dict = {}
    for disease, syms in disease_to_symptoms.items():
        for sym in syms:
            symptom_to_diseases.setdefault(sym, [])
            if disease not in symptom_to_diseases[sym]:
                symptom_to_diseases[sym].append(disease)

    # Load canonical→expressions map from lexicon.csv
    raw_lexicon = _load_lexicon(lexicon_path)

    # Build frozenset lexicon: frozenset(lemma_tokens) → canonical_symptom
    lexicon: dict = {}
    for canonical, expressions in raw_lexicon.items():
        for expr in expressions:
            if not expr:
                continue
            token_set = _lemmatize_expression(expr)
            if token_set:
                lexicon[token_set] = canonical

    return disease_to_symptoms, symptom_to_diseases, lexicon


# Load everything once at import time
print("[lexicon_branch] Loading data structures...")
disease_to_symptoms, symptom_to_diseases, lexicon = _build_structures(
    _DISEASE_PATHS, _LEXICON_PATH
)
print(f"[lexicon_branch] Loaded {len(disease_to_symptoms)} diseases, "
      f"{len(symptom_to_diseases)} symptoms, {len(lexicon)} lexicon entries.")


# ── Stage 2 Functions ────────────────────────────────────────────────────────

def compute_idf_weights(s_to_d: dict = None) -> dict:
    """
    Compute IDF-like specificity weight per symptom.
    Symptoms shared across more diseases get lower weight.
    idf(s) = 1 - (num_diseases_with_s / total_diseases)
    """
    s_to_d = s_to_d or symptom_to_diseases
    total = len(set(d for ds in s_to_d.values() for d in ds))
    if total == 0:
        return {s: 1.0 for s in s_to_d}
    return {s: 1.0 - len(ds) / total for s, ds in s_to_d.items()}


def match_symptoms(token_bag: set, lex: dict = None) -> list:
    """
    Match patient token bag against lexicon entries.
    A match occurs when ALL tokens of an expression are present in the token bag.
    Returns list of matched canonical symptom names (deduplicated).
    """
    lex = lex or lexicon
    return list(set(canon for ts, canon in lex.items() if ts.issubset(token_bag)))


def lexicon_branch(
    token_bag: set,
    d_to_s: dict = None,
    s_to_d: dict = None,
    lex: dict = None,
) -> dict:
    """
    Stage 2 — Lexicon Branch.

    Args:
        token_bag: Set of lemmatized tokens from Stage 1.
        d_to_s: disease_to_symptoms dict (uses module-level default).
        s_to_d: symptom_to_diseases dict (uses module-level default).
        lex: lexicon frozenset dict (uses module-level default).

    Returns:
        {
          'Diabetes': {
            'matched_symptoms': [{'canonical': str, 'specificity': float}, ...],
            'idf_score': float   # normalized weighted score [0, 1]
          },
          'Skin Cancer': { ... }
        }
    """
    d_to_s = d_to_s or disease_to_symptoms
    s_to_d = s_to_d or symptom_to_diseases
    lex    = lex    or lexicon

    idf     = compute_idf_weights(s_to_d)
    matched = match_symptoms(token_bag, lex)

    # Total IDF weight across all matched symptoms (for normalization)
    all_w = sum(idf.get(s, 0.0) for s in matched)

    results = {}
    for disease, d_syms in d_to_s.items():
        d_matched = [s for s in matched if s in d_syms]
        d_w = sum(idf.get(s, 0.0) for s in d_matched)
        results[disease] = {
            "matched_symptoms": [
                {"canonical": s, "specificity": round(idf.get(s, 0.0), 4)}
                for s in d_matched
            ],
            "idf_score": d_w / all_w if all_w > 0 else 0.0,
        }

    return results
