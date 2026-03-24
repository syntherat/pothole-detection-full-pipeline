import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from pathlib import Path


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Data" / "synthetic_pothole_dataset.csv"
MODEL_PATH = BASE_DIR / "Data" / "pothole_ai_model.pkl"


# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv(DATASET_PATH)


# -----------------------------
# Features and labels
# -----------------------------

X = data[[
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "speed"
]]

y = data["label"]


# -----------------------------
# Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Train AI model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Save model
# -----------------------------

joblib.dump(model, MODEL_PATH)


# -----------------------------
# Evaluate model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAI Model Performance:\n")
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy*100:.2f}%")