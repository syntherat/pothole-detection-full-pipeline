"""
Runs the physics detector plus the AI filter over the synthetic dataset and
scores the result.

Revised 2026-09-06 (item 4.2 of PAVE_v2_change_log_REVIEWED.docx):
scoring is now **per event**, not per row.

Why that matters: each pothole occupies twelve consecutive labelled rows, so
row-wise scoring counted one pothole as twelve detections and one miss as twelve
misses. A detector whose job is to find discrete objects has to be scored in
units of those objects. The row-wise figures are still printed, clearly marked,
because the two numbers together say more than either alone.

The same correction was already applied on the integration side -- see
group_label_events() in integration/test_integration.py, which defines an event
the same way this does.

Usage:
    python Model/run_detector_on_dataset.py
    python Model/run_detector_on_dataset.py --rolling --model Data/pothole_ai_model_rolling.pkl
    python Model/run_detector_on_dataset.py --no-plot
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

# This script lives in Model/ but PotholeDetector lives in detector_py/, so the
# module is not importable by bare name from here. Same sys.path pattern the
# integration adapters use -- see integration/sensor_adapter.py.
DETECTOR_DIR = BASE_DIR / "detector_py"
sys.path.insert(0, str(DETECTOR_DIR))

from pothole_detection import PotholeDetector  # noqa: E402
import features as feat  # noqa: E402

DATASET_PATH = BASE_DIR / "Data" / "synthetic_pothole_dataset.csv"
MODEL_PATH = BASE_DIR / "Data" / "pothole_ai_model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--rolling", action="store_true",
                        help="Build the rolling features. MUST match how the model was trained.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(DATASET_PATH)
    data = feat.ensure_event_ids(data)

    ai_model = joblib.load(args.model)

    if args.rolling:
        data = feat.add_rolling_features(data)
        feature_columns = feat.full_feature_columns()
    else:
        feature_columns = list(feat.RAW_FEATURE_COLUMNS)

    # A feature-set mismatch between training and evaluation shows up as a wrong
    # score rather than an error, so check it rather than trust the flag.
    expected = getattr(ai_model, "n_features_in_", None)
    if expected is not None and expected != len(feature_columns):
        raise SystemExit(
            f"Model expects {expected} features but {len(feature_columns)} were built. "
            f"Pass --rolling if the model was trained with it (or drop it if not)."
        )

    detector = PotholeDetector()
    predictions = ai_model.predict(data[feature_columns])

    # --- row-wise tallies, kept only for contrast --------------------------
    row_tp = row_fp = row_fn = 0

    # --- event-wise tallies, the number that means something ---------------
    true_event_ids = set(data.loc[data["event_id"] >= 0, "event_id"].astype(int))
    detected_event_ids: set[int] = set()
    false_positive_rows = 0
    physics_detections = 0
    ai_confirmed_detections = 0
    detected_times: list[float] = []

    event_id_col = data["event_id"].to_numpy()
    label_col = data["label"].to_numpy()

    for i, row in enumerate(data.itertuples(index=False)):
        result = detector.process_sample(
            timestamp=row.timestamp,
            ax=row.ax, ay=row.ay, az=row.az,
            gx=row.gx, gy=row.gy, gz=row.gz,
            speed=row.speed,
        )

        prediction = predictions[i]
        actual = label_col[i]
        eid = int(event_id_col[i])

        if result["pothole_detected"]:
            physics_detections += 1
            if prediction == 1:
                ai_confirmed_detections += 1
                detected_times.append(row.timestamp)
                # The cascade fires once, on the row where the IMPACT lands, which
                # is inside the labelled window. Credit the event it belongs to;
                # a firing outside every window is a genuine false positive.
                if eid >= 0:
                    detected_event_ids.add(eid)
                else:
                    false_positive_rows += 1

        if prediction == 1 and actual == 1:
            row_tp += 1
        elif prediction == 1 and actual == 0:
            row_fp += 1
        elif prediction == 0 and actual == 1:
            row_fn += 1

    tp = len(detected_event_ids & true_event_ids)
    fn = len(true_event_ids - detected_event_ids)
    recall = tp / len(true_event_ids) if true_event_ids else 0.0
    precision = tp / (tp + false_positive_rows) if (tp + false_positive_rows) else 0.0

    print("\nDetection Summary")
    print("------------------------")
    print("Physics detections (rows):", physics_detections)
    print("AI confirmed detections (rows):", ai_confirmed_detections)

    print("\nEVENT-LEVEL METRICS  <- the meaningful unit")
    print("-------------------------------------------")
    print(f"Labelled pothole events : {len(true_event_ids)}")
    print(f"Events detected         : {tp}")
    print(f"Events missed           : {fn}")
    print(f"Spurious firings        : {false_positive_rows}  (cascade fired outside every labelled window)")
    print(f"Recall                  : {recall * 100:.2f}%")
    print(f"Precision               : {precision * 100:.2f}%")

    print("\nRow-level metrics (for contrast only -- NOT a detector score)")
    print("-------------------------------------------------------------")
    print(f"True Positives : {row_tp}")
    print(f"False Positives: {row_fp}")
    print(f"False Negatives: {row_fn}")
    print("Each pothole spans ~12 rows, so these count one object up to twelve")
    print("times. They describe the AI filter's per-row behaviour, not how many")
    print("potholes the system actually found.")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(data["timestamp"], data["az"], label="Vertical Acceleration (az)")
        for t in detected_times:
            plt.axvline(t, alpha=0.4)
        plt.title("Pothole Detection Visualization")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Acceleration (m/s²)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
