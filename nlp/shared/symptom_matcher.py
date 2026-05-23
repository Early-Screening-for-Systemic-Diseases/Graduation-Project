# nlp/shared/symptom_matcher.py

from .preprocessing import clean_text
from .lemmatizer import lemmatize_text

def match_symptoms(patient_text: str, lexicon: dict) -> list:
    """
    Match patient text to canonical symptoms using the lexicon.

    Args:
        patient_text (str): Raw patient free-text
        lexicon (dict): canonical_symptom -> list of cleaned & lemmatized expressions

    Returns:
        List of canonical symptoms mentioned in patient text
    """
    if not patient_text or not lexicon:
        return []

    # 1. Clean and lemmatize patient text
    cleaned_text = clean_text(patient_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    
    # 2. Split patient text into a set of words for matching
    patient_tokens = set(lemmatized_text.split())
    
    detected_symptoms = []

    # 3. Check each canonical symptom
    for canonical, expressions in lexicon.items():
        for expr in expressions:
            expr_tokens = set(expr.split())
            # If all words in expression are in patient text → match
            if expr_tokens.issubset(patient_tokens):
                detected_symptoms.append(canonical)
                break  # Avoid duplicates

    return detected_symptoms
