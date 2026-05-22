# nlpv2/pipeline/preprocess.py
# Stage 1 — Preprocessing
# Produces: token_bag (set of lemmas) and clean_sentence (str)
# Models loaded once at module startup.

import unicodedata
import spacy
import subprocess
import sys

try:
    import contractions as _contractions
    _has_contractions = True
except ImportError:
    _has_contractions = False

# ── Load SpaCy once at module startup ───────────────────────────────────────
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    _nlp = spacy.load("en_core_web_sm")


def preprocess(raw_text: str) -> dict:
    """
    Stage 1 — Preprocessing.

    Args:
        raw_text: The raw patient input string.

    Returns:
        {
            'token_bag': set of lemmatized tokens (punctuation and spaces excluded),
            'clean_sentence': lowercased, contraction-expanded, unicode-normalized string
        }
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # 1. Normalize unicode (handles curly quotes, special dashes, etc.)
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Lowercase
    text = text.lower()

    # 3. Expand contractions (I'm → I am, don't → do not, etc.)
    if _has_contractions:
        try:
            text = _contractions.fix(text)
        except Exception:
            pass

    clean_sentence = text

    # 4. SpaCy tokenization + lemmatization to build token bag
    doc = _nlp(text)
    token_bag = set(
        t.lemma_.lower()
        for t in doc
        if not t.is_punct and not t.is_space and t.lemma_.strip()
    )

    return {"token_bag": token_bag, "clean_sentence": clean_sentence}
