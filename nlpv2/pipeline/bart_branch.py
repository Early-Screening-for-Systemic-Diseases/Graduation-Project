from transformers import pipeline

# Load once at module startup
nli = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

TEMPERATURE = 1.5

def bart_branch(clean_sentence, diseases):
    labels = [f'This person shows signs of {d}' for d in diseases]
    
    # multi_label=False as required by the spec
    out = nli(clean_sentence, candidate_labels=labels, multi_label=False)
    
    results = {}
    for label, score in zip(out['labels'], out['scores']):
        d = label.replace('This person shows signs of ', '')
        results[d] = {
            'raw_score': float(score),
            'calibrated_score': float(score) ** (1.0 / TEMPERATURE)
        }
        
    return results