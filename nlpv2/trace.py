import os
import json
from pipeline.data_loader import load_knowledge_base
from pipeline.preprocess import preprocess
from pipeline.lexicon_branch import lexicon_branch, compute_idf_weights
from pipeline.biobert_branch import biobert_branch, build_disease_embeddings
from pipeline.bart_branch import bart_branch
from pipeline.fusion import entropy_fusion
from pipeline.spacy_correction import spacy_post_correct, detect_negated_symptoms, detect_intensity

def trace_pipeline(raw_text: str):
    print("="*80)
    print(f"PIPELINE TRACE FOR: \"{raw_text}\"")
    print("="*80)

    # 0. Load Data
    print("\n[0] LOADING KNOWLEDGE BASE...")
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/lexicon.csv')
    symptom_to_diseases, disease_to_symptoms, lexicon = load_knowledge_base(DATA_PATH)
    disease_embeddings, symptom_embeddings = build_disease_embeddings(disease_to_symptoms)
    diseases = list(disease_to_symptoms.keys())
    idf = compute_idf_weights(symptom_to_diseases)
    print(f"Loaded {len(diseases)} diseases and {len(symptom_to_diseases)} canonical symptoms.")

    # 1. Preprocess
    print("\n[1] PREPROCESSING...")
    prep = preprocess(raw_text)
    token_bag = prep['token_bag']
    clean = prep['clean_sentence']
    print(f"Clean Sentence: '{clean}'")
    print(f"Token Bag (Lemmas): {sorted(list(token_bag))}")

    # 2. Lexicon Branch
    print("\n[2] LEXICON BRANCH...")
    lex_res = lexicon_branch(token_bag, disease_to_symptoms, symptom_to_diseases, lexicon)
    for d, res in lex_res.items():
        matched = [s['canonical'] for s in res['matched_symptoms']]
        print(f"  -> {d}: matched {matched}, IDF Score = {res['idf_score']:.4f}")

    # 3. BioBERT Branch
    print("\n[3] BIOBERT BRANCH...")
    bio_res = biobert_branch(clean, disease_embeddings, symptom_embeddings)
    for d, res in bio_res.items():
        print(f"  -> {d}: Cosine Similarity = {res['similarity']:.4f}")

    # 4. BART Branch
    print("\n[4] BART ZERO-SHOT BRANCH...")
    bart_res = bart_branch(clean, diseases)
    for d, res in bart_res.items():
        print(f"  -> {d}: Raw Prob = {res['raw_score']:.4f}, Calibrated = {res['calibrated_score']:.4f}")

    # 5. Fusion
    print("\n[5] ENTROPY FUSION...")
    fused = entropy_fusion(lex_res, bio_res, bart_res, diseases)
    # Print the weights for the first disease (weights are the same across diseases for the branches)
    first_d = diseases[0]
    weights = fused[first_d]['weights']
    print(f"  -> Branch Weights calculated via Inverse Entropy:")
    print(f"       Lexicon Weight : {weights['lexicon']:.4f}")
    print(f"       BioBERT Weight : {weights['biobert']:.4f}")
    print(f"       BART Weight    : {weights['bart']:.4f}")
    for d, res in fused.items():
        print(f"  -> {d} FUSED SCORE: {res['fused_score']:.4f}")

    # 6. SpaCy Correction (Negation & Intensity)
    print("\n[6] SPACY CORRECTION (NEGATION & INTENSITY)...")
    negated = detect_negated_symptoms(clean, list(idf.keys()))
    intensity = detect_intensity(clean)
    mult = {'HIGH': 1.3, 'MILD': 0.8, 'NONE': 1.0}[intensity]
    print(f"  -> Detected Negated Symptoms: {list(negated)}")
    print(f"  -> Detected Intensity: {intensity} (Multiplier = {mult}x)")
    
    final = spacy_post_correct(fused, lex_res, bio_res, clean, idf)
    for d, res in final.items():
        score_before = fused[d]['fused_score']
        score_after_subtraction = score_before
        
        # Show exactly what got subtracted
        lex_syms = [s['canonical'] for s in lex_res[d].get('matched_symptoms', [])]
        bio_syms = [s['canonical'] for s in bio_res[d].get('matched_symptoms', [])]
        lex_w_total = sum(idf.get(s, 0.0) for s in lex_syms)
        bio_w_total = sum(idf.get(s, 0.0) for s in bio_syms)
        
        subtractions = []
        for sym in negated:
            if sym in lex_syms and lex_w_total > 0:
                drop = res['weights']['lexicon'] * (idf.get(sym, 0.0) / lex_w_total)
                subtractions.append(f"Lexicon penalty for '{sym}' (-{drop:.4f})")
                score_after_subtraction -= drop
            if sym in bio_syms and bio_w_total > 0:
                drop = res['weights']['biobert'] * (idf.get(sym, 0.0) / bio_w_total)
                subtractions.append(f"BioBERT penalty for '{sym}' (-{drop:.4f})")
                score_after_subtraction -= drop
        
        print(f"\n  [{d}] Math Breakdown:")
        print(f"      Base Fused Score : {score_before:.4f}")
        for sub in subtractions:
            print(f"      {sub}")
        print(f"      After Subtraction: {max(0.0, score_after_subtraction):.4f}")
        print(f"      Apply Multiplier : * {mult}")
        print(f"      FINAL CAPPED     : {res['final_score']:.4f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    sentences = [
        # Case 1: Healthy / Normal (Testing NO INDICATION handling)
        "I just have a mild headache and feel a bit tired from working late.",
        
        # Case 2: Extreme Diabetes Symptoms (Testing MILD/HIGH Intensity Multipliers)
        "I have severe and extreme constant thirst, and I've experienced sudden massive weight loss.",
        
        # Case 3: Negated Skin Cancer (Testing Negation Subtraction)
        "I do not have a sore that won't heal, but I noticed a new spot getting bigger."
    ]
    
    for s in sentences:
        trace_pipeline(s)
