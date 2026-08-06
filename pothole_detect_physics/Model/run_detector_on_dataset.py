import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pothole_detection import PotholeDetector
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
# Load trained AI model
# -----------------------------

ai_model = joblib.load(MODEL_PATH)


# -----------------------------
# Initialize physics detector
# -----------------------------

detector = PotholeDetector()


physics_detections = 0
ai_confirmed_detections = 0

true_positives = 0
false_positives = 0
false_negatives = 0


detected_times = []

print("\nRunning pothole detection with AI filtering...\n")


# -----------------------------
# Run AI predictions ONCE
# -----------------------------

X = data[["ax", "ay", "az", "gx", "gy", "gz", "speed"]]
predictions = ai_model.predict(X)


# -----------------------------
# Process dataset
# -----------------------------

for i, row in data.iterrows():

    result = detector.process_sample(
        timestamp=row["timestamp"],
        ax=row["ax"],
        ay=row["ay"],
        az=row["az"],
        gx=row["gx"],
        gy=row["gy"],
        gz=row["gz"],
        speed=row["speed"]
    )

    prediction = predictions[i]
    actual = row["label"]


    # -----------------------------
    # Physics detection
    # -----------------------------

    if result["pothole_detected"]:

        physics_detections += 1

        if prediction == 1:

            ai_confirmed_detections += 1
            detected_times.append(row["timestamp"])

            print("Pothole confirmed at time:", row["timestamp"])
            print(result)
            print()


    # -----------------------------
    # Evaluation metrics
    # -----------------------------

    if prediction == 1 and actual == 1:
        true_positives += 1

    elif prediction == 1 and actual == 0:
        false_positives += 1

    elif prediction == 0 and actual == 1:
        false_negatives += 1


# -----------------------------
# Accuracy calculation
# -----------------------------

total = true_positives + false_positives + false_negatives
accuracy = true_positives / total if total > 0 else 0


# -----------------------------
# Summary
# -----------------------------

print("\nDetection Summary")
print("------------------------")
print("Physics detections:", physics_detections)
print("AI confirmed detections:", ai_confirmed_detections)


print("\nEvaluation Metrics")
print("-------------------")
print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
print(f"Accuracy: {accuracy*100:.2f}%")


# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(12,6))

plt.plot(data["timestamp"], data["az"], label="Vertical Acceleration (az)")

for t in detected_times:
    plt.axvline(t, alpha=0.4)

plt.title("Pothole Detection Visualization")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()

plt.show()