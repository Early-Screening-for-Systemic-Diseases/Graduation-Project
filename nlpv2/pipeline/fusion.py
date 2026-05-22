import math

BRANCH_THRESHOLD = 0.50
MUTED_WEIGHT     = 0.05

def normalize_scores(scores: dict) -> dict:
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    # If the confidence gap between diseases is less than 3%, the branch has no strong opinion.
    # We map this to 0.0 so it fails the BRANCH_THRESHOLD gate and gets muted.
    if (hi - lo) < 0.03:
        return { k: 0.0 for k in scores }
    return { k: (v - lo) / (hi - lo) for k, v in scores.items() }

def branch_passes_gate(scores: dict) -> bool:
    return max(scores.values()) >= BRANCH_THRESHOLD

def shannon_entropy(scores: dict) -> float:
    return -sum(p * math.log(p + 1e-9) for p in scores.values() if p > 0)

def entropy_to_trust(H: float) -> float:
    return 1.0 / (1.0 + H)

def entropy_fusion(lex_results, bio_results, bart_results, diseases):
    # Extract raw scores, but apply ABSOLUTE GATES.
    # If a branch is just guessing random noise, we force its score to 0.0.
    
    # BioBERT MUST have found at least one specific symptom >= 0.55 similarity
    raw_bio = {d: bio_results[d]['similarity'] if len(bio_results[d]['matched_symptoms']) > 0 else 0.0 for d in diseases}
    
    # BART MUST be highly confident (e.g., > 0.75) to be considered a real signal
    raw_bart = {d: bart_results[d]['calibrated_score'] if bart_results[d]['calibrated_score'] > 0.75 else 0.0 for d in diseases}
    
    # Lexicon is already mathematically strict (0.0 if no match)
    raw_lex = {d: lex_results[d]['idf_score'] for d in diseases}

    # Normalize score dicts
    lex_scores  = normalize_scores(raw_lex)
    bio_scores  = normalize_scores(raw_bio)
    bart_scores = normalize_scores(raw_bart)

    # Gate 1: mute branches with no signal
    lex_trust  = entropy_to_trust(shannon_entropy(lex_scores))  if branch_passes_gate(lex_scores)  else MUTED_WEIGHT
    bio_trust  = entropy_to_trust(shannon_entropy(bio_scores))  if branch_passes_gate(bio_scores)  else MUTED_WEIGHT
    bart_trust = entropy_to_trust(shannon_entropy(bart_scores)) if branch_passes_gate(bart_scores) else MUTED_WEIGHT

    total = lex_trust + bio_trust + bart_trust
    w_lex, w_bio, w_bart = lex_trust/total, bio_trust/total, bart_trust/total

    fused = {}
    for d in diseases:
        score = (w_lex * lex_scores[d]) + (w_bio * bio_scores[d]) + (w_bart * bart_scores[d])
        fused[d] = { 
            'fused_score': score, 
            'weights': {'lexicon': w_lex, 'biobert': w_bio, 'bart': w_bart},
            'component_scores': {'lexicon': lex_scores[d], 'biobert': bio_scores[d], 'bart': bart_scores[d]} 
        }
        
    return fused