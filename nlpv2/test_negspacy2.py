import os
import spacy
from spacy.tokens import Token
from negspacy.negation import Negex
import pandas as pd

if not Token.has_extension('negex'):
    Token.set_extension('negex', default=False)

nlp = spacy.load('en_core_web_sm')

# Build patterns
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/lexicon.csv')
df = pd.read_csv(DATA_PATH)
patterns = []
for _, row in df.iterrows():
    symptom = row['symptom_name'].strip()
    # Add symptom name itself
    patterns.append({"label": "SYMPTOM", "pattern": symptom.replace("’", "'"), "id": symptom})
    if pd.notna(row['expressions']):
        expressions = [expr.strip() for expr in str(row['expressions']).split(',')]
        for expr in expressions:
            patterns.append({"label": "SYMPTOM", "pattern": expr.replace("’", "'"), "id": symptom})

ruler = nlp.add_pipe('entity_ruler', before='ner')
ruler.add_patterns(patterns)
nlp.add_pipe('negex')

doc = nlp("I don't have a sore that won't heal, but I do have a new spot on my skin that is getting bigger.")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Negated: {ent._.negex}")
