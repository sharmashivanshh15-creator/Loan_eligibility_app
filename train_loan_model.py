import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("loan_applications.csv")

df.columns = df.columns.str.strip()
print("\nDataset loaded:", df.shape)
print("Columns:", df.columns)

df["decision"] = df["decision"].astype(str).str.strip()
print("\nUnique decision values before mapping:", df["decision"].unique())

feature_cols = [
    "income",
    "expenses",
    "credit_score",
    "emis",
    "loan_requested",
    "tenure_years",
    "dti",
    "emi_ratio",
]

x = df[feature_cols]

label_map = {"Rejected": 0, "Borderline": 1, "Approved": 2}
y = df["decision"].map(label_map)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
)

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Rejected", "Borderline", "Approved"]
    )
)

with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved loan_model.pkl and scaler.pkl")