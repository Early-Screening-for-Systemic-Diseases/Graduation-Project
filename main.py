import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Load dataset
df = pd.read_csv(r"D:\Survey_T_Dataset\diabetes_012_health_indicators_BRFSS2015.csv")

# 2️⃣ Binary label
df["label"] = df["Diabetes_012"].apply(lambda x: 0 if x == 0 else 1)
y = df["label"]

# 3️⃣ Features
features = [
    'BMI', 'HighBP', 'HighChol', 'CholCheck',
    'Smoker', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age', 'Education', 'Income'
]

X = df[features]

# 4️⃣ One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 5️⃣ Save columns for API
joblib.dump(X.columns, "model_columns.pkl")

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7️⃣ Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight={0:1, 1:3},
    random_state=42
)

model.fit(X_train, y_train)

# 8️⃣ Save model
joblib.dump(model, "diabetes_survey_model.pkl")

print("✅ Training finished. Model saved.")

