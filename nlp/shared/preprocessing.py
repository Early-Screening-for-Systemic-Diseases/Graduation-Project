# nlp/shared/preprocessing.py

import re
try:
    import emoji
except Exception:
    emoji = None

try:
    import ftfy
except Exception:
    ftfy = None

try:
    import contractions
except Exception:
    contractions = None

try:
    from unidecode import unidecode
except Exception:
    unidecode = None

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
    
    if ftfy is not None:
        try:
            text = ftfy.fix_text(text)
        except Exception:
            pass
    if contractions is not None:
        try:
            text = contractions.fix(text)
        except Exception:
            pass
    if unidecode is not None:
        try:
            text = unidecode(text)
        except Exception:
            pass
    text = text.lower()
    if emoji is not None:
        try:
            text = emoji.replace_emoji(text, replace="")
        except Exception:
            # emoji library present but failed; fall back to simple removal
            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    else:
        # emoji package not installed — remove high unicode chars as a fallback
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
