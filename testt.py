from nlp.shared.disease_matcher_full import SemanticMatcher

disease_symptoms = {"diabetes": {"fatigue"}, "anemia": {"tiredness"}}
matcher = SemanticMatcher(disease_symptoms)

text = "I feel very tired all the time"
results = matcher.match(text)
print(results)