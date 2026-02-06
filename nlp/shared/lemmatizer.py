# nlp/shared/lemmatizer.py

import spacy

nlp_spacy = spacy.load("en_core_web_sm")

def lemmatize_text(text: str) -> str:
    """
    Lemmatize already-cleaned text using POS-aware lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    doc = nlp_spacy(text)
    lemmas = []

    for token in doc:
        if token.is_space:
            continue
        
        lemma = token.lemma_.lower()

        # Ignore spaCy failures
        if lemma == token.text.lower():
            lemma = token.text.lower()

        if lemma == "-pron-":
            lemma = token.text.lower()

        lemmas.append(lemma)

    return " ".join(lemmas)
