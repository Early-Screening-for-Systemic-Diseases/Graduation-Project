from zero_shot_symptom_matcher import ZeroShotSymptomMatcher

matcher = ZeroShotSymptomMatcher(
    "nlp/data/symptom_type_map.csv",
    threshold=0.4
)

tests = [
    "I get worn out after small activities",
    "I need to stop to catch my breath",
    "Bleeding does not stop after small cuts",
    "I feel thirsty all the time and pee a lot",
    "I feel weak and dizzy"
]

for t in tests:
    print("\nTEXT:", t)
    results = matcher.match(t)

    if not results:
        print("→ No symptoms detected")
    else:
        for r in results:
            print(f"→ {r['symptom']} | {r['type']} | score={r['score']}")
