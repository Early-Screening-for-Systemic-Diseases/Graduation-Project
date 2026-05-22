import os
from .data_loader import load_knowledge_base
from .preprocess import preprocess
from .lexicon_branch import lexicon_branch, compute_idf_weights
from .biobert_branch import biobert_branch, build_disease_embeddings
from .bart_branch import bart_branch
from .fusion import entropy_fusion
from .spacy_correction import spacy_post_correct

# ==========================================
# MODULE STARTUP: Load data and build globals
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/lexicon.csv')
symptom_to_diseases, disease_to_symptoms, lexicon = load_knowledge_base(DATA_PATH)
disease_embeddings, symptom_embeddings = build_disease_embeddings(disease_to_symptoms)
# ==========================================
# 11. Confidence Level Helper
# ==========================================
def confidence_level(score: float) -> str:
    if score > 0.75:  return 'HIGH'
    if score > 0.50:  return 'MEDIUM'
    if score > 0.30:  return 'LOW'
    return 'NO INDICATION'

# ==========================================
# 12. Main Pipeline
# ==========================================
def run_pipeline(raw_text: str) -> dict:
    prep = preprocess(raw_text)
    token_bag, clean = prep['token_bag'], prep['clean_sentence']
    diseases = list(disease_to_symptoms.keys())

    # Run the three parallel branches 
    lex_res  = lexicon_branch(token_bag, disease_to_symptoms, symptom_to_diseases, lexicon)
    bio_res  = biobert_branch(clean, disease_embeddings, symptom_embeddings)
    bart_res = bart_branch(clean, diseases)

    # Fuse and correct
    fused = entropy_fusion(lex_res, bio_res, bart_res, diseases)
    idf   = compute_idf_weights(symptom_to_diseases)
    final = spacy_post_correct(fused, lex_res, bio_res, clean, idf)

    # Rank results
    ranked = sorted(final.items(), key=lambda x: x[1]['final_score'], reverse=True)
    
    return {
        'input': raw_text,
        'ranked_diseases': [{
            'disease':          d,
            'final_score':      round(data['final_score'], 4),
            'confidence_level': confidence_level(data['final_score']),
            'lexicon_matched':  [s['canonical'] for s in lex_res[d]['matched_symptoms']],
            'biobert_sim':      round(bio_res[d]['similarity'], 4),
            'bart_calibrated':  round(bart_res[d]['calibrated_score'], 4),
            'fusion_weights':   data['weights'],
            'negated':          data['negated_symptoms'],
            'intensity':        data['intensity']
        } for d, data in ranked]
    }