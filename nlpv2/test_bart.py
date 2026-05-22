import os
from pipeline.bart_branch import bart_branch

diseases = ["Diabetes", "Skin Cancer"]
sentences = [
    "I have a headache that comes and goes, and my lower back hurts after sitting for a long time. Nothing else unusual.",
    "I feel a bit stressed lately because of work and I haven't been sleeping well, but I eat normally and feel physically fine.",
    "My knee hurts when I climb stairs and I feel stiff in the morning. My doctor said it might be early arthritis."
]

for s in sentences:
    res = bart_branch(s, diseases)
    print(f"\nSentence: {s}")
    for d, scores in res.items():
        print(f"{d}: Calibrated = {scores['calibrated_score']:.4f}")
