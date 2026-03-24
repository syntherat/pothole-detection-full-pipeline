import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------------
# Constants
# -----------------------------

g = 9.81
sampling_rate = 400
dt = 1 / sampling_rate


# -----------------------------
# Simulation length
# -----------------------------

total_time = 200
samples = int(total_time * sampling_rate)

timestamps = np.arange(0, total_time, dt)


# -----------------------------
# Normal driving noise
# -----------------------------

ax = np.random.normal(0, 0.2, samples)
ay = np.random.normal(0, 0.2, samples)
az = np.random.normal(g, 0.2, samples)

gx = np.random.normal(0, 0.02, samples)
gy = np.random.normal(0, 0.02, samples)
gz = np.random.normal(0, 0.02, samples)

speed = np.random.uniform(10, 18, samples)


# -----------------------------
# Labels
# -----------------------------

labels = np.zeros(samples)


# -----------------------------
# Insert pothole events
# -----------------------------

num_potholes = 200

for _ in range(num_potholes):

    start = np.random.randint(500, samples - 500)

    drop = np.random.uniform(4, 7)
    freefall = np.random.uniform(0.3, 1.0)
    impact = np.random.uniform(18, 28)

    az[start:start+5] = drop
    az[start+5:start+10] = freefall
    az[start+10:start+12] = impact

    az[start:start+12] += np.random.normal(0, 0.3, 12)

    labels[start:start+12] = 1


# -----------------------------
# Insert speed breaker events
# -----------------------------

num_speedbreakers = 150

for _ in range(num_speedbreakers):

    start = np.random.randint(500, samples - 500)

    rise = np.random.uniform(12, 16)
    fall = np.random.uniform(6, 8)

    az[start:start+10] = rise
    az[start+10:start+20] = fall

    az[start:start+20] += np.random.normal(0, 0.3, 20)

    # IMPORTANT: label stays 0 (not a pothole)


# -----------------------------
# Create dataset
# -----------------------------

data = pd.DataFrame({
    "timestamp": timestamps,
    "ax": ax,
    "ay": ay,
    "az": az,
    "gx": gx,
    "gy": gy,
    "gz": gz,
    "speed": speed,
    "label": labels
})


# -----------------------------
# Save dataset
# -----------------------------

data.to_csv(BASE_DIR / "Data" / "synthetic_pothole_dataset.csv", index=False)


print("Dataset generated successfully!")
print("Total samples:", samples)
print("Potholes inserted:", num_potholes)
print("Speed breakers inserted:", num_speedbreakers)