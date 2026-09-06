"""
Loops the CSV, calls each stage in order, drives the whole cascade
python integration/orchestrator.py
"""

import argparse
import uuid
from pathlib import Path
import pandas as pd

from schema import PotholeCandidateEvent
from sensor_adapter import SensorSession
from vision_adapter import VisionSession
from frame_provider import MockFrameProvider
from fusion import should_trigger_vision, fuse, VISION_THRESHOLD
from pave_connector import send_to_pave

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "pothole_detect_physics" / "Data" / "synthetic_pothole_dataset.csv"


def run(limit: int | None = None,
        dataset_path: Path | None = None,
        frame_provider=None):
    """
    `frame_provider` is anything with get_frame(timestamp, event_id) and
    get_gps(timestamp) -- MockFrameProvider by default, CarlaFrameProvider for a
    recorded run. Both return None when they cannot supply a value, and those
    rows are counted and reported rather than silently dropped.
    """
    data = pd.read_csv(dataset_path or DATASET_PATH)
    if limit:
        data = data.head(limit)

    provider = frame_provider if frame_provider is not None else MockFrameProvider()

    sensor = SensorSession()
    vision = VisionSession()

    confirmed_events = []
    skipped_no_frame = 0
    skipped_no_gps = 0

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

        # --- bridge: frame + GPS from whichever provider is in use ---
        frame_path = provider.get_frame(event.timestamp, event.event_id)
        if frame_path is None:
            skipped_no_frame += 1
            continue
        event.frame_path = frame_path

        gps = provider.get_gps(event.timestamp)
        if gps is None:
            # CarlaFrameProvider returns None outside the recorded GNSS window.
            # Publishing an event with no position would put a marker at (0, 0).
            skipped_no_gps += 1
            continue
        event.lat, event.lng = gps

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
    if skipped_no_gps:
        print(f"Sensor-triggered rows skipped for want of a GPS fix: {skipped_no_gps}")
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
