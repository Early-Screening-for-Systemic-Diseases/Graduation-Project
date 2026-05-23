import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent / 'nlp' / 'data'

for name in ['diabetes_symptoms.csv','anemia_symptoms.csv']:
    p = base / name
    print('\n==', name, '==')
    df = pd.read_csv(p)
    print('Columns:', list(df.columns))
    for idx, row in df.iterrows():
        disease = row['disease'] if 'disease' in row else row.iloc[0]
        # collect non-null symptom cells
        symptoms = [s for s in row[1:] if pd.notna(s)]
        print('\nRow', idx, 'disease:', disease)
        print('Raw symptom cells count:', len(symptoms))
        print('Symptom cells:', symptoms)
        # normalize: lower + strip
        norm = {str(s).strip().lower() for s in symptoms}
        print('Normalized symptom set size:', len(norm))
        print('Normalized set sample:', list(norm)[:10])

print('\nDone')
