import pandas as pd
import spacy

# Load SpaCy once for lemmatizing the lexicon expressions
nlp = spacy.load('en_core_web_sm')

def load_knowledge_base(csv_path: str):
    df = pd.read_csv(csv_path)
    
    symptom_to_diseases = {}
    disease_to_symptoms = {}
    lexicon = {}
    
    for _, row in df.iterrows():
        symptom = row['symptom_name'].strip()
        disease = row['disease_name'].strip()
        
        # Build symptom_to_diseases [cite: 49, 50, 51, 52, 53]
        if symptom not in symptom_to_diseases:
            symptom_to_diseases[symptom] = []
        if disease not in symptom_to_diseases[symptom]:
            symptom_to_diseases[symptom].append(disease)
            
        # Build disease_to_symptoms [cite: 54, 55, 56, 57]
        if disease not in disease_to_symptoms:
            disease_to_symptoms[disease] = []
        if symptom not in disease_to_symptoms[disease]:
            disease_to_symptoms[disease].append(symptom)
            
        # Build lexicon [cite: 58, 59, 60, 61, 62]
        # All expressions must be lemmatized using the same SpaCy model as Stage 1 [cite: 63]
        if pd.notna(row['expressions']):
            expressions = [expr.strip() for expr in str(row['expressions']).split(',')]
            for expr in expressions:
                doc = nlp(expr.lower())
                lemmatized_tokens = frozenset([t.lemma_ for t in doc if not t.is_punct and not t.is_space])
                if lemmatized_tokens:
                    lexicon[lemmatized_tokens] = symptom
                    
    return symptom_to_diseases, disease_to_symptoms, lexicon