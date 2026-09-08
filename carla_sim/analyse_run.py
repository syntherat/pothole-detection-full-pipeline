"""
carla_sim/analyse_run.py

Runs the sensor stage over a recorded CARLA run and reports what it found.

    venv/Scripts/python.exe carla_sim/analyse_run.py carla_sim/out/run_<ts>

Reports physics-FSM detections against the ground truth the recorder wrote, plus
the az extremes -- because on Level A the useful diagnosis was never the
detection count on its own, but whether az even entered the FSM's DROP and
IMPACT windows (see issue #35 and the tuning table in config.py).

READ THE NUMBERS WITH THE CONTROL IN HAND. A Level B tile is a hole behind a
ramp of the same height, so a detection may be the lip rather than the cavity.
Record a `--control` run over the same route and diff the two; a detection rate
quoted without that comparison does not separate the two causes.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

sys.path.insert(0, str(BASE_DIR / "pothole_detect_physics" / "detector_py"))
from pothole_detection import PotholeDetector  # noqa: E402

# Driven directly rather than through integration.sensor_adapter.SensorSession.
# SensorSession.check_row also calls the RandomForest's predict_proba once PER ROW,
# building a one-row DataFrame each time -- ~40 min for a 24k-row run, and this
# script never reads ai_score. The FSM below is byte-identical either way.

# PotholeDetector's gates, quoted here so the diagnosis below is self-contained.
G = 9.81
DROP_BELOW = 6.81
IMPACT_ABOVE = 19.81


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    sensors = pd.read_csv(run_dir / "sensors.csv")

    import json
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    truth = json.loads((run_dir / "ground_truth.json").read_text(encoding="utf-8"))

    print(f"run   : {run_dir.name}")
    print(f"level : {meta.get('level')}")
    print(f"tiles : {meta.get('tiles_spawned')}  control_run={meta.get('control_run')}")
    print(f"rows  : {len(sensors)}   labelled positive: {int((sensors['label'] == 1).sum())}")
    print(f"truth : {truth.get('hit_count')} of {truth.get('placed_count')} potholes driven over")

    # --- az envelope: did the signal even reach the FSM's windows? ---
    az = sensors["az"]
    in_drop = int((az < DROP_BELOW).sum())
    in_impact = int((az > IMPACT_ABOVE).sum())
    print(f"\naz    : min {az.min():.2f}  max {az.max():.2f}  (rest G = {G})")
    print(f"        samples below DROP gate ({DROP_BELOW}):   {in_drop}")
    print(f"        samples above IMPACT gate ({IMPACT_ABOVE}): {in_impact}")

    # --- run the real detector ---
    detector = PotholeDetector()
    detections = []
    # itertuples, not iterrows: the FSM is stateful so order matters, but building a
    # Series per row does not.
    cols = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "speed"]
    for r in sensors[cols].itertuples(index=False, name=None):
        ts, ax_, ay_, az_, gx_, gy_, gz_, sp = r
        if detector.process_sample(
            timestamp=ts, ax=ax_, ay=ay_, az=az_, gx=gx_, gy=gy_, gz=gz_, speed=sp
        )["pothole_detected"]:
            detections.append(ts)

    print(f"\nFSM detections: {len(detections)}")
    for t in detections[:20]:
        # Was there a labelled pothole near this detection?
        near = sensors[(sensors["timestamp"] - t).abs() < 0.25]
        labelled = int((near["label"] == 1).any())
        print(f"   t={t:8.3f}s   near_labelled_pothole={bool(labelled)}")

    if not detections:
        print("\n   No detections. Before tuning anything, check the az envelope above:")
        print("   if nothing crossed the DROP gate, the wheel never fell far enough,")
        print("   and no threshold change to the FSM will help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
