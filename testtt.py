from pathlib import Path
from nlp.shared.predictor import predict

# Example texts
texts = [
    "I feel extremely tired and weak",
    "I have high blood sugar and frequent urination",
    "Walking makes me exhausted very quickly",
    "I am constantly thirsty and urinate frequently"
]

# Run predictions
for text in texts:
    output = predict(text)
    print(f"Input: {text}")
    print(f"Lexicon matched: {output['lexicon_matched']}")
    print("Results:")
    results = output.get('results')
    results_map = output.get('results_map')

    # If detailed list is available, print disease, matched symptoms, and percentage
    if isinstance(results, list) and results:
        for r in results:
            disease = r.get('disease', '<unknown>')
            pct = r.get('percentage')
            matched = r.get('matched_symptoms') or []
            if pct is not None:
                print(f"  → Disease: {disease} | Symptoms: {matched} | % matched: {pct}")
            else:
                print(f"  → Disease: {disease} | Symptoms: {matched} | % matched: N/A")
    # Else fall back to results_map (dict) which contains percentage and matched_symptoms
    elif isinstance(results_map, dict) and results_map:
        for disease, info in results_map.items():
            pct = info.get('percentage')
            matched = info.get('matched_symptoms') or []
            print(f"  → Disease: {disease} | Symptoms: {matched} | % matched: {pct}")
    else:
        print(results)
    print("="*40)