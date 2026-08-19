"""
Loops the CSV, calls each stage in order, drives the whole cascade
python integration/orchestrator.py
"""

import uuid
from pathlib import Path
import pandas as pd

from schema import PotholeCandidateEvent
from sensor_adapter import SensorSession
from vision_adapter import VisionSession
from frame_provider import get_mock_frame, simulate_gps
from fusion import should_trigger_vision, fuse, VISION_THRESHOLD
from pave_connector import send_to_pave

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "pothole_detect_physics" / "Data" / "synthetic_pothole_dataset.csv"


def run(limit: int | None = None):
    data = pd.read_csv(DATASET_PATH)
    if limit:
        data = data.head(limit)

    sensor = SensorSession()
    vision = VisionSession()

    confirmed_events = []

    for _, row in data.iterrows():
        event = PotholeCandidateEvent(
            event_id=str(uuid.uuid4()),
            timestamp=float(row["timestamp"]),
        )

        # --- stage 1: sensor ---
        sensor_result = sensor.check_row(row)
        event.physics_detected = sensor_result["physics_detected"]
        event.physics_depth_estimate = sensor_result["physics_depth_estimate"]
        event.physics_length_estimate = sensor_result["physics_length_estimate"]
        event.ai_score = sensor_result["ai_score"]
        event.sensor_triggered = should_trigger_vision(
            event.physics_detected, event.ai_score
        )

        if not event.sensor_triggered:
            continue

        # --- bridge: mock frame + GPS until real data exists ---
        event.frame_path = get_mock_frame(event.event_id)
        event.lat, event.lng = simulate_gps(event.timestamp)

        # --- stage 2: vision confirmation ---
        vision_result = vision.confirm(event.frame_path, conf=VISION_THRESHOLD)
        event.vision_score = vision_result["vision_score"]
        event.vision_confirmed = vision_result["vision_confirmed"]

        # --- fusion ---
        event.final_decision, event.final_confidence = fuse(
            event.ai_score, event.vision_score
        )

        print(
            f"[{event.timestamp:.2f}s] sensor={event.ai_score:.2f} "
            f"vision={event.vision_score:.2f} -> "
            f"{'CONFIRMED' if event.final_decision else 'rejected'} "
            f"(confidence={event.final_confidence:.2f})"
        )

        if event.final_decision:
            send_to_pave(event)
            confirmed_events.append(event)

    print(f"\nTotal confirmed pothole events: {len(confirmed_events)}")
    if skipped_no_frame:
        print(f"Sensor-triggered rows skipped for want of a frame: {skipped_no_frame}")
        if not confirmed_events:
            print(
                "  Every triggered row was skipped, so nothing could reach the vision\n"
                "  stage. With the mock provider this means no stand-in images were\n"
                "  found -- see pothole_detection_app/data/sample_images/README.md."
            )
    return confirmed_events


def main():
    parser = argparse.ArgumentParser(description="Run the sensor-vision cascade.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows. The full synthetic "
                             "dataset is 80,000 rows with a YOLO pass per trigger.")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Sensor CSV in contract #1 shape. Defaults to the synthetic dataset.")
    parser.add_argument("--carla-run", type=Path, default=None,
                        help="Recorded CARLA run directory. Uses real time-synced "
                             "frames and GNSS instead of the mocks.")
    args = parser.parse_args()

    provider = None
    if args.carla_run:
        from carla_frame_provider import CarlaFrameProvider
        provider = CarlaFrameProvider(args.carla_run)
        print(f"Using CARLA run: {args.carla_run}")

    if args.limit is None:
        print("No --limit given: processing the ENTIRE dataset. "
              "Ctrl-C and pass --limit 2000 if that was not intended.")

    run(limit=args.limit, dataset_path=args.dataset, frame_provider=provider)


if __name__ == "__main__":
    main()
