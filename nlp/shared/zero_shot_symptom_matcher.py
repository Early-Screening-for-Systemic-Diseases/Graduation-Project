import pandas as pd
from transformers import pipeline

class ZeroShotSymptomMatcher:
    def __init__(self, csv_path, threshold=0.4):
        self.threshold = threshold

        df = pd.read_csv(csv_path)
        if not {"Symptom", "Type"}.issubset(df.columns):
            raise ValueError("CSV must contain Symptom and Type columns")

        self.symptoms = df["Symptom"].tolist()
        self.symptom_to_type = dict(zip(df["Symptom"], df["Type"]))

        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def match(self, text):
        hypotheses = [f"The patient has {s}." for s in self.symptoms]

        result = self.classifier(
            text,
            candidate_labels=hypotheses,
            multi_label=True
        )

        matches = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= self.threshold:
                symptom = label.replace("The patient has ", "").rstrip(".")
                matches.append({
                    "symptom": symptom,
                    "type": self.symptom_to_type.get(symptom),
                    "score": round(score, 3)
                })

        return sorted(matches, key=lambda x: x["score"], reverse=True)
