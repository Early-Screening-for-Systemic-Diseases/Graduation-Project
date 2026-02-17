# Diabetes Survey API (Railway deploy)

Place your trained model `diabetes_gb_balanced_model.pkl` in this folder. Optionally add `model_columns.pkl` (pickled list of training feature columns).

To generate `model_columns.pkl` (if the model exposes `feature_names_in_`):

```bash
python -m pip install -r requirements.txt
python generate_columns.py
```

Run locally:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Deploy to Railway (CLI):

```bash
railway login
cd questionnaire_diabetes
railway init
railway up
```
