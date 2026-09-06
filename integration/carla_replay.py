"""
integration/carla_replay.py

Replays a recorded CARLA run onto the map dashboard: drives the car marker along
the route in time order and drops pothole markers as they are detected.

WHY THIS EXISTS
---------------
CARLA mode could draw the town and place markers, but showed no vehicle and
never raised a proximity alert -- issue #28. The position was always there,
recorded in gnss.csv every tick; nothing published it to the UI.

Browser geolocation cannot fill that gap: it would put the car in Bhopal while
the CARLA roads sit near (0, 0), thousands of km apart.

WHAT IT PUBLISHES, AND WHAT IT DOES NOT CLAIM
---------------------------------------------
Markers come from the **sensor stage only** -- the physics FSM plus the
RandomForest filter, i.e. `sensor_triggered`. They are labelled that way.

They are deliberately NOT labelled "hybrid (sensor+vision)". CARLA Level A
defines potholes as data plus a physics impulse; there is no hole in the road
mesh, so the dash camera records clean tarmac and the vision stage confirms
nothing. Publishing vision-confirmed markers from a Level A run would be a
fabricated result. See context/15-carla-testbed-plan.md.

Rule 3: this is an adapter in integration/. It writes contract #12
(vehicle_position.json) and goes through pave_connector for contract #7 rather
than writing the events file itself.

Usage:
    python integration/carla_replay.py carla_sim/out/run_20260906_195639
    python integration/carla_replay.py <run> --speed 4 --reset --loop
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402

from sensor_adapter import SensorSession, FEATURE_COLUMNS  # noqa: E402
from fusion import should_trigger_vision  # noqa: E402
from pave_connector import send_record_to_pave, EVENTS_FILE  # noqa: E402

POSITION_FILE = BASE_DIR / "vehicle_position.json"

# The UI polls; publishing at the recorded 400 Hz would rewrite the file 400
# times a second for no visible benefit. 20 Hz is smoother than any poll
# interval the dashboard uses.
PUBLISH_HZ = 20.0


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compass bearing from one fix to the next, for pointing the car icon."""
    d_lng = math.radians(lng2 - lng1)
    y = math.sin(d_lng) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d_lng))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def load_gnss(run_dir: Path) -> list[tuple[float, float, float]]:
    path = run_dir / "gnss.csv"
    if not path.exists():
        raise SystemExit(f"No gnss.csv in {run_dir} -- is this a recorded run directory?")
    fixes: list[tuple[float, float, float]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fixes.append((float(row["timestamp"]),
                          float(row["latitude"]),
                          float(row["longitude"])))
    fixes.sort(key=lambda r: r[0])
    return fixes


def detect_sensor_events(run_dir: Path) -> list[dict]:
    """
    Sensor-stage detections, with the timestamp each one fired at.

    Runs the same SensorSession and the same should_trigger_vision() gate the
    orchestrator uses, so a marker here means exactly what a Stage-1 trigger
    means there -- no second definition of "detected".
    """
    sensors = run_dir / "sensors.csv"
    if not sensors.exists():
        raise SystemExit(f"No sensors.csv in {run_dir}")

    data = pd.read_csv(sensors)
    session = SensorSession()
    events: list[dict] = []

    # SensorSession.check_row() builds a one-row DataFrame and calls
    # predict_proba once PER ROW. That is ~25 minutes for a 40,000-row run, which
    # is intolerable for something a person waits on before a replay starts.
    #
    # The classifier is stateless per row, so its scores are computed here in a
    # single vectorised call. The FSM is NOT -- it is a state machine and must
    # still see every sample in order.
    #
    # This uses the session's own detector and model rather than constructing
    # them separately, so there is still exactly one definition of "detected":
    # same PotholeDetector, same pickle, same should_trigger_vision() gate.
    scores = session.ai_model.predict_proba(data[FEATURE_COLUMNS])[:, 1]

    for i, row in enumerate(data.itertuples(index=False)):
        physics = session.physics_detector.process_sample(
            timestamp=row.timestamp,
            ax=row.ax, ay=row.ay, az=row.az,
            gx=row.gx, gy=row.gy, gz=row.gz,
            speed=row.speed,
        )
        ai_score = float(scores[i])
        if should_trigger_vision(physics["pothole_detected"], ai_score):
            events.append({
                "timestamp": float(row.timestamp),
                "ai_score": ai_score,
                "depth": physics.get("depth_estimate"),
            })
    return events


def nearest_fix(fixes: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    """Linear scan is fine: this runs once per detection, not once per tick."""
    best = min(fixes, key=lambda r: abs(r[0] - t))
    return best[1], best[2]


def publish_position(lat: float, lng: float, heading: float, t: float,
                     progress: float, run_name: str) -> None:
    POSITION_FILE.write_text(json.dumps({
        "lat": lat,
        "lng": lng,
        "heading": heading,
        "timestamp": t,
        "progress": progress,
        "run": run_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="carla_sim/out/run_<timestamp>")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback rate. 1.0 = the recorded pace, 4.0 = four times faster.")
    parser.add_argument("--reset", action="store_true",
                        help="Clear pave_events.json first, so the map shows only this run.")
    parser.add_argument("--loop", action="store_true", help="Restart when the run ends.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    fixes = load_gnss(run_dir)
    if len(fixes) < 2:
        raise SystemExit("Need at least two GNSS fixes to replay a route.")

    print(f"Run: {run_dir.name}")
    print(f"GNSS fixes: {len(fixes):,}  spanning {fixes[0][0]:.2f}s .. {fixes[-1][0]:.2f}s")

    print("Scoring the sensor stage over the recorded run ...")
    events = detect_sensor_events(run_dir)
    print(f"Sensor-stage detections: {len(events)}")

    if args.reset and EVENTS_FILE.exists():
        EVENTS_FILE.unlink()
        print("Cleared pave_events.json")

    t_start, t_end = fixes[0][0], fixes[-1][0]
    duration = (t_end - t_start) / max(args.speed, 0.01)
    print(f"Replaying over {duration:.1f}s of wall clock (speed {args.speed}x). Ctrl-C to stop.\n")

    step = 1.0 / PUBLISH_HZ

    while True:
        pending = list(events)
        wall0 = time.time()
        idx = 0
        heading = 0.0

        while True:
            elapsed = (time.time() - wall0) * args.speed
            t = t_start + elapsed
            if t >= t_end:
                break

            while idx + 1 < len(fixes) and fixes[idx + 1][0] <= t:
                idx += 1
            lat, lng = fixes[idx][1], fixes[idx][2]

            if idx + 1 < len(fixes):
                nxt = fixes[idx + 1]
                if (nxt[1], nxt[2]) != (lat, lng):
                    heading = bearing_deg(lat, lng, nxt[1], nxt[2])

            progress = (t - t_start) / (t_end - t_start)
            publish_position(lat, lng, heading, t, progress, run_dir.name)

            while pending and pending[0]["timestamp"] <= t:
                ev = pending.pop(0)
                e_lat, e_lng = nearest_fix(fixes, ev["timestamp"])
                send_record_to_pave({
                    "event_id": str(uuid.uuid4()),
                    "lat": e_lat,
                    "lng": e_lng,
                    "confidence": ev["ai_score"],
                    # Honest provenance: Level A has no visible pothole, so this
                    # is the sensor stage alone. Never "hybrid (sensor+vision)".
                    "detected_by": "sensor (CARLA replay)",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"  [{ev['timestamp']:7.2f}s] pothole marker at "
                      f"{e_lat:.6f}, {e_lng:.6f}  (score {ev['ai_score']:.2f})")

            time.sleep(step)

        print(f"\nReplay finished. {len(events) - len(pending)} markers published.")
        if not args.loop:
            break
        print("Looping.\n")


if __name__ == "__main__":
    main()
