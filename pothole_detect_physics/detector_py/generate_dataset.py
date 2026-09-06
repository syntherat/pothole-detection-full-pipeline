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

# Added 2026-09-06. This generator was previously unseeded, which meant the
# committed synthetic_pothole_dataset.csv could not be reproduced and the model
# pickled from it could not be retrained on the same data. Seeding fixes that
# going forward; it does NOT recover the committed file, which was generated
# before the seed existed and is therefore not reproducible by any seed.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


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

# event_id / event_type make an event the unit of measurement rather than the
# row. Without them a train/test split cuts through the middle of a single
# pothole (leakage) and scoring counts one pothole as twelve detections.
# Consumed by Model/features.py, Model/train_ai_model.py and
# Model/run_detector_on_dataset.py.
#
# -1 means "no event here". Speed breakers get an id too even though their label
# stays 0: they are real events that must not be split across train and test
# either, and distinguishing them from plain background is what event_type is
# for.
event_id = np.full(samples, -1, dtype=np.int64)
event_type = np.full(samples, "none", dtype=object)
next_event_id = 0


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
    # Overlapping events overwrite earlier ids, matching how az itself is
    # overwritten above -- the later event is what the signal actually shows.
    event_id[start:start+12] = next_event_id
    event_type[start:start+12] = "pothole"
    next_event_id += 1


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

    # Speed breakers are written AFTER potholes at independent random starts, so
    # they sometimes land on top of one. When that happens `az` becomes the speed
    # breaker's signal, but the pothole's label used to survive underneath it --
    # leaving rows labelled "pothole" that actually contain the dataset's own
    # designated hard negative. Measured on a fresh 80k generation: 100 rows,
    # 4.2 % of all positive labels, mean az 10.02 against 6.41 for clean pothole
    # rows. The label must follow the signal.
    labels[start:start+20] = 0

    event_id[start:start+20] = next_event_id
    event_type[start:start+20] = "speed_breaker"
    next_event_id += 1

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
    "label": labels,
    "event_id": event_id,
    "event_type": event_type
})


# -----------------------------
# Save dataset
# -----------------------------

data.to_csv(BASE_DIR / "Data" / "synthetic_pothole_dataset.csv", index=False)


print("Dataset generated successfully!")
print("Total samples:", samples)
print("Potholes inserted:", num_potholes)
print("Speed breakers inserted:", num_speedbreakers)
print("Distinct events tagged:", next_event_id)
print(f"Seed: {RANDOM_SEED} (reproducible)")