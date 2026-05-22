import os
import spacy
from spacy.tokens import Token
from negspacy.negation import Negex
import pandas as pd

# Explicitly register the negex extension on Token before adding the pipe.
if not Token.has_extension('negex'):
    Token.set_extension('negex', default=False)

# Load correction model
nlp_correction = spacy.load('en_core_web_sm')

# Add EntityRuler so negspacy can recognize symptoms as entities
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/lexicon.csv')
df = pd.read_csv(DATA_PATH)
patterns = []

for _, row in df.iterrows():
    symptom = row['symptom_name'].strip()
    
    # Build a lemma pattern for the canonical symptom name
    symptom_lemmas = [{"LEMMA": t.lemma_.lower()} for t in nlp_correction(symptom) if not t.is_punct and not t.is_space]
    if symptom_lemmas:
        patterns.append({"label": "SYMPTOM", "pattern": symptom_lemmas, "id": symptom})
    
    # Build lemma patterns for all expressions
    if pd.notna(row['expressions']):
        expressions = [expr.strip() for expr in str(row['expressions']).split(',')]
        for expr in expressions:
            expr_lemmas = [{"LEMMA": t.lemma_.lower()} for t in nlp_correction(expr) if not t.is_punct and not t.is_space]
            if expr_lemmas:
                patterns.append({"label": "SYMPTOM", "pattern": expr_lemmas, "id": symptom})

ruler = nlp_correction.add_pipe('entity_ruler', before='ner')
ruler.add_patterns(patterns)

# Add negex pipe globally
nlp_correction.add_pipe('negex')

HIGH_WORDS = {'extreme', 'severe', 'intense', 'chronic', 'constant', 'terrible', 'horrible'}
MILD_WORDS = {'mild', 'slight', 'occasional', 'minor', 'little', 'bit', 'sometimes'}

def detect_negated_symptoms(clean_sentence, all_symptoms):
    doc = nlp_correction(clean_sentence)
    negated = set()
    for ent in doc.ents:
        if ent.label_ == "SYMPTOM" and ent._.negex:
            negated.add(ent.ent_id_)
    return negated

def detect_intensity(clean_sentence):
    doc = nlp_correction(clean_sentence)
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & HIGH_WORDS: return 'HIGH'
    if lemmas & MILD_WORDS: return 'MILD'
    return 'NONE'

def spacy_post_correct(fused, lex_results, bio_results, clean_sentence, idf_weights):
    negated   = detect_negated_symptoms(clean_sentence, list(idf_weights.keys()))
    intensity = detect_intensity(clean_sentence)
    mult = {'HIGH': 1.3, 'MILD': 0.8, 'NONE': 1.0}[intensity]
    
    corrected = {}
    for disease, res in fused.items():
        score = res['fused_score']
        lex_syms = [s['canonical'] for s in lex_results[disease].get('matched_symptoms', [])]
        bio_syms = [s['canonical'] for s in bio_results[disease].get('matched_symptoms', [])]
        
        lex_w_total = sum(idf_weights.get(s, 0.0) for s in lex_syms)
        bio_w_total = sum(idf_weights.get(s, 0.0) for s in bio_syms)
        
        for sym in negated:
            if sym in lex_syms and lex_w_total > 0:
                score = max(0.0, score - res['weights']['lexicon'] * (idf_weights.get(sym, 0.0) / lex_w_total))
            if sym in bio_syms and bio_w_total > 0:
                score = max(0.0, score - res['weights']['biobert'] * (idf_weights.get(sym, 0.0) / bio_w_total))
                
        # Apply intensity multiplier and cap at 1.0
        score = min(1.0, score * mult)
        
        corrected[disease] = { 
            **res, 
            'final_score': score,
            'negated_symptoms': list(negated), 
            'intensity': intensity, 
            'multiplier': mult 
        }
        
    return corrected
