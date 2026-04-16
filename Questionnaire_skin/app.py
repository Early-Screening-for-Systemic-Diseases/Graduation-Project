import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
model           = joblib.load("skin_cancer_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
target_map      = joblib.load("target_map.pkl")
inv_target_map  = {v: k for k, v in target_map.items()}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Skin Cancer Risk API",
    description="Predicts skin cancer risk (Low Risk / High Risk / Moderate Risk) from a patient questionnaire.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request schema — 26 questions
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    age: int
    skin_tone: str                        # Very Fair / Fair / Olive / Brown / Dark
    natural_hair_color: str               # Black / Red / Blonde / Light Brown / Dark Brown
    freckles: str                         # None / Few / Moderate / Many
    number_of_moles: str                  # 0-10 / 11-25 / 26-50 / 51-100 / More than 100
    moles_larger_than_5mm: str            # None / 1-2 / 3-5 / More than 5
    changing_mole_or_lesion: str          # Yes / No
    new_unusual_skin_growth: str          # Yes / No
    skin_easily_sunburned: str            # Never / Rarely / Sometimes / Often / Always
    tanning_ability: str                  # Tans easily and deeply / Tans gradually / Rarely tans / Never tans
    sunburns_in_childhood: str            # None / 1-2 / 3-5 / More than 5
    severe_blistering_sunburns_ever: str  # Yes / No
    daily_sun_exposure_hours: str         # Less than 1 hour / 1-3 hours / 3-6 hours / More than 6 hours
    outdoor_work_or_recreation: str       # Rarely / Sometimes / Often / Daily
    use_of_sunscreen: str                 # Always (SPF 30+) / Usually (SPF 15-29) / Sometimes / Rarely / Never
    use_of_tanning_bed: str               # Never / Occasionally / Regularly (monthly) / Frequently (weekly)
    years_of_tanning_bed_use: str         # Never used / Less than 1 year / 1-5 years / More than 5 years
    personal_history_skin_cancer: str     # Yes / No
    family_history_skin_cancer: str       # None / One relative / Two or more relatives
    immunosuppressive_medication: str     # Yes / No
    radiation_therapy_history: str        # Yes / No
    geographic_location_uv: str           # Low UV region / Moderate UV region / High UV region
    years_lived_in_high_uv_area: str      # Never / Less than 5 years / 5-15 years / More than 15 years
    perform_skin_self_checks: str         # Monthly / Every few months / Once a year / Rarely / Never
    seen_dermatologist: str               # More than once a year / Annually / Once every few years / Never
    diet_high_antioxidants: str           # Often / Sometimes / Rarely / Never

# ---------------------------------------------------------------------------
# Encoding — mirrors the notebook's encode_new_patient exactly
# ---------------------------------------------------------------------------
ORDINAL_ENCODERS = {
    'skin_easily_sunburned':       ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'tanning_ability':             ['Tans easily and deeply', 'Tans gradually', 'Rarely tans', 'Never tans'],
    'daily_sun_exposure_hours':    ['Less than 1 hour', '1-3 hours', '3-6 hours', 'More than 6 hours'],
    'years_of_tanning_bed_use':    ['Never used', 'Less than 1 year', '1-5 years', 'More than 5 years'],
    'years_lived_in_high_uv_area': ['Never', 'Less than 5 years', '5-15 years', 'More than 15 years'],
    'number_of_moles':             ['0-10', '11-25', '26-50', '51-100', 'More than 100'],
    'sunburns_in_childhood':       ['None', '1-2', '3-5', 'More than 5'],
    'geographic_location_uv':      ['Low UV region', 'Moderate UV region', 'High UV region'],
    'freckles':                    ['None', 'Few', 'Moderate', 'Many'],
    'moles_larger_than_5mm':       ['None', '1-2', '3-5', 'More than 5'],
    'family_history_skin_cancer':  ['None', 'One relative', 'Two or more relatives'],
    'outdoor_work_or_recreation':  ['Rarely', 'Sometimes', 'Often', 'Daily'],
    'use_of_tanning_bed':          ['Never', 'Occasionally', 'Regularly (monthly)', 'Frequently (weekly)'],
    'perform_skin_self_checks':    ['Monthly', 'Every few months', 'Once a year', 'Rarely', 'Never'],
    'seen_dermatologist':          ['More than once a year', 'Annually', 'Once every few years', 'Never'],
    'diet_high_antioxidants':      ['Often', 'Sometimes', 'Rarely', 'Never'],
}

BINARY_FIELDS = [
    'changing_mole_or_lesion', 'new_unusual_skin_growth',
    'severe_blistering_sunburns_ever', 'personal_history_skin_cancer',
    'immunosuppressive_medication', 'radiation_therapy_history',
]

NOMINAL_FIELDS = ['skin_tone', 'natural_hair_color', 'use_of_sunscreen']


def encode_patient(patient: PatientInput) -> pd.DataFrame:
    row = pd.DataFrame([patient.model_dump()])

    for col, cats in ORDINAL_ENCODERS.items():
        if col in row.columns:
            enc = OrdinalEncoder(
                categories=[cats],
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )
            row[col] = enc.fit_transform(row[[col]])
            row[col] = row[col].clip(lower=0)

    for col in BINARY_FIELDS:
        if col in row.columns:
            row[col] = row[col].apply(
                lambda v: 1.0 if str(v).strip().lower() == 'yes' else 0.0
            )

    row = pd.get_dummies(row, columns=[c for c in NOMINAL_FIELDS if c in row.columns], drop_first=False)

    bool_cols = row.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        row[bool_cols] = row[bool_cols].astype(float)

    row = row.reindex(columns=feature_columns, fill_value=0).astype(float)
    return row

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Skin Cancer Risk API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientInput):
    row            = encode_patient(patient)
    predicted_code = model.predict(row)[0]
    proba          = model.predict_proba(row)[0]

    probabilities = {
        inv_target_map[i]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }

    return {
        "risk_level": inv_target_map[predicted_code],
        "probabilities": probabilities,
    }
