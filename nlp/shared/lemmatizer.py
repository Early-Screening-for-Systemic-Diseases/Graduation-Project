# import spacy
# from spacy.cli import download

# Try to load the model, download if not found
import spacy
import subprocess
import sys

try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
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
