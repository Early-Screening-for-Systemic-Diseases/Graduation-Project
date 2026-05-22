def compute_idf_weights(symptom_to_diseases):
    """
    Penalizes symptoms that appear across many diseases.
    """
    total = len(set(d for ds in symptom_to_diseases.values() for d in ds))
    return { s: 1.0 - len(ds)/total for s, ds in symptom_to_diseases.items() }

def match_symptoms(token_bag, lexicon):
    """
    Matches frozensets of lemmatized patient expressions against the patient's token bag.
    """
    return list(set(canon for ts, canon in lexicon.items() if ts.issubset(token_bag)))

def lexicon_branch(token_bag, disease_to_symptoms, symptom_to_diseases, lexicon):
    idf = compute_idf_weights(symptom_to_diseases)
    matched = match_symptoms(token_bag, lexicon)
    all_w = sum(idf.get(s, 0.0) for s in matched)
    
    results = {}
    for disease, d_syms in disease_to_symptoms.items():
        d_matched = [s for s in matched if s in d_syms]
        d_w = sum(idf.get(s, 0.0) for s in d_matched)
        
        results[disease] = {
            'matched_symptoms': [{'canonical': s, 'specificity': idf.get(s, 0.0)} for s in d_matched],
            'idf_score': d_w / all_w if all_w > 0 else 0.0
        }
        
    return results