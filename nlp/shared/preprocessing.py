# nlp/shared/preprocessing.py

import re
import emoji
import ftfy
import contractions
from unidecode import unidecode

def clean_text(text: str) -> str:
    """
    Preprocess medical free-text:
    1. Fix messy Unicode
    2. Expand contractions
    3. Normalize Unicode to ASCII
    4. Lowercase
    5. Remove emojis
    6. Remove punctuation
    7. Remove extra spaces
    """
    if not isinstance(text, str):
        return ""
    
    text = ftfy.fix_text(text)
    text = contractions.fix(text)
    text = unidecode(text)
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
