import unicodedata
import contractions
import spacy

# Load model outside the function to adhere to critical rules [cite: 258]
nlp = spacy.load('en_core_web_sm')

def preprocess(raw_text: str) -> dict:
    """
    Returns:
    { 'token_bag': set,  'clean_sentence': str } [cite: 69]
    """
    # Normalize and clean [cite: 72, 73, 74]
    text = unicodedata.normalize('NFKC', raw_text)
    text = text.lower()
    text = contractions.fix(text)
    clean_sentence = text
    
    # Create lemmatized token bag [cite: 76, 77]
    doc = nlp(text)
    token_bag = set(t.lemma_ for t in doc if not t.is_punct and not t.is_space)
    
    return { 'token_bag': token_bag, 'clean_sentence': clean_sentence }