import joblib, os, traceback
p = os.path.join(os.getcwd(), 'diabetes_gb_balanced_model.pkl')
print('CWD:', os.getcwd())
print('Path exists:', os.path.exists(p), p)
try:
    m = joblib.load(p)
    print('Loaded model type:', type(m))
    for attr in ['feature_names_in_', 'n_features_in_']:
        if hasattr(m, attr):
            print(attr, getattr(m, attr))
except Exception as e:
    print('Load error:')
    traceback.print_exc()
