# nlp/shared/lexicon_loader.py

import pandas as pd
from .preprocessing import clean_text
from .lemmatizer import lemmatize_text

def load_lexicon(path: str) -> dict:
    """
    Load lexicon CSV and convert into canonical_symptom -> list of lemma expressions
    """
    if path is None:
        # Assume lexicon.csv lives in the same folder as this file's parent → nlp/data/
        from pathlib import Path
        file_location = Path(__file__).resolve()
        project_root = file_location.parent.parent.parent  # go up from shared/ → project root
        path = project_root / "data" / "lexicon.csv"

    
    lexicon_df = pd.read_csv(path, sep=",", encoding="cp1252")
    
    # Remove empty columns
    lexicon_df = lexicon_df.dropna(axis=1, how="all")
    
    # Fill empty cells
    lexicon_df = lexicon_df.fillna("")
    
    # Clean column names
    lexicon_df.columns = lexicon_df.columns.str.strip().str.replace('\ufeff', '', regex=True)
    
    lexicon_dict = {}
    
    for _, row in lexicon_df.iterrows():
        canonical = row["Canonical Symptom"].strip()
        
        # Merge all patient expressions
        expressions = [clean_text(x) for x in row[1:] if x != ""]
        
        # Lemmatize each expression
        expressions = [lemmatize_text(x) for x in expressions]
        
        lexicon_dict[canonical] = expressions
    
    return lexicon_dict
