import joblib
import sys
import os

MODEL_FILENAME = "diabetes_gb_balanced_model.pkl"
OUT = "model_columns.pkl"

if not os.path.exists(MODEL_FILENAME):
    print(f"Model {MODEL_FILENAME} not found in current folder.")
    sys.exit(1)

model = joblib.load(MODEL_FILENAME)

if hasattr(model, "feature_names_in_"):
    cols = list(model.feature_names_in_)
    joblib.dump(cols, OUT)
    print(f"Wrote {OUT} with {len(cols)} columns.")
else:
    print("Model has no 'feature_names_in_'. If you trained with a DataFrame -> try exporting the columns list from training.")
    sys.exit(2)
import joblib
import sys
import os

MODEL_FILENAME = "diabetes_gb_balanced_model.pkl"
OUT = "model_columns.pkl"

if not os.path.exists(MODEL_FILENAME):
    print(f"Model {MODEL_FILENAME} not found in current folder.")
    sys.exit(1)

model = joblib.load(MODEL_FILENAME)

if hasattr(model, "feature_names_in_"):
    cols = list(model.feature_names_in_)
    joblib.dump(cols, OUT)
    print(f"Wrote {OUT} with {len(cols)} columns.")
else:
    print("Model has no 'feature_names_in_'. If you trained with a DataFrame -> try exporting the columns list from training.")
    sys.exit(2)