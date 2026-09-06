"""
carla_sim/scenario/drive_and_record.py

Level A main loop. Drives a fixed route, seeds a jerk at each pothole, and
records everything the existing pipeline needs.

Output (see context/20-data-contracts.md):

    out/run_<ts>/
      sensors.csv       timestamp,ax,ay,az,gx,gy,gz,speed,label   <- contract #1, EXACTLY
      gnss.csv          timestamp,frame,latitude,longitude,altitude
      frame_index.csv   frame,timestamp,path                      <- closes issue #18
      frames/<frame>.png
      ground_truth.json pothole positions + which were actually hit
      run_meta.json     config snapshot, for reproducibility

sensors.csv is column-identical to synthetic_pothole_dataset.csv on purpose:
the FSM, the RandomForest, fusion and the connector all run against it unchanged.

Usage:
    python carla_sim/scenario/drive_and_record.py
    python carla_sim/scenario/drive_and_record.py --no-camera --ticks 40000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

try:
    import carla
except ImportError:
    sys.exit("carla module not found. `pip install carla`, and start the simulator.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                    # noqa: E402
from scenario import potholes as ph              # noqa: E402
from scenario import route as rt                 # noqa: E402
from scenario.impulse import ImpulseApplier, wheel_positions   # noqa: E402
from scenario.sensors import SensorRig           # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Contract #1. Do not reorder, do not extend -- see Rule 4.
SENSOR_COLUMNS = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "speed", "label"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA Level A pothole recorder.")
    p.add_argument("--town", default=config.TOWN)
    p.add_argument("--ticks", type=int, default=config.MAX_TICKS)
    p.add_argument("--potholes", type=int, default=config.NUM_POTHOLES)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--no-camera", action="store_true", help="IMU only -- much faster")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def make_run_dir(explicit: Path | None) -> Path:
    run_dir = explicit or (config.OUT_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    args = parse_args()

    if config.WHEEL_POSITION_SCALE is None:
        logger.error(
            "config.WHEEL_POSITION_SCALE is not set.\n"
            "Run `python carla_sim/verify_setup.py` first and paste in the measured value.\n"
            "Guessing this wrong means wheel-over detection silently never fires."
        )
        return 1

    if config.IMU_GRAVITY_AT_REST is None:
        logger.warning(
            "config.IMU_GRAVITY_AT_REST is not set -- the FSM's az thresholds are "
            "unverified for this build. Recording anyway, but check verify_setup.py."
        )

    run_dir = make_run_dir(args.out)
    logger.info("Run directory: %s", run_dir)

    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT_S)

    world = client.load_world(args.town) if args.town else client.get_world()
    original_settings = world.get_settings()

    vehicle = None
    rig = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
        settings.substepping = config.SUBSTEPPING
        settings.max_substep_delta_time = config.MAX_SUBSTEP_DELTA_TIME
        settings.max_substeps = config.MAX_SUBSTEPS
        world.apply_settings(settings)

        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        spawn = spawn_points[args.seed % len(spawn_points)]

        blueprint = world.get_blueprint_library().find(config.VEHICLE_BLUEPRINT)
        vehicle = world.spawn_actor(blueprint, spawn)
        logger.info("Spawned %s", config.VEHICLE_BLUEPRINT)

        # Let the suspension settle before anything is recorded, otherwise the
        # first hundred samples are the car dropping onto its springs.
        for _ in range(200):
            world.tick()

        route = rt.build_route(world_map, spawn, config.ROUTE_LENGTH_M)
        placed = ph.place_along_route(
            route, args.potholes, config.POTHOLE_SPACING_M,
            config.POTHOLE_RADIUS_M, config.POTHOLE_LATERAL_OFFSET_M,
            config.POTHOLE_SEVERITY_RANGE, args.seed,
        )

        tracker = ph.PotholeTracker(placed, config.LABEL_WINDOW_S)
        applier = ImpulseApplier(
            vehicle, config.IMPULSE_DELTA_V, config.IMPULSE_TICKS,
            unload_ticks=config.UNLOAD_TICKS,
            unload_scale=config.UNLOAD_FORCE_SCALE,
        )
        follower = rt.WaypointFollower(vehicle, route, config.TARGET_SPEED_KMH)

        if args.no_camera:
            config.CAMERA_ENABLED = False
        rig = SensorRig(world, vehicle, config)

        sensors_path = run_dir / "sensors.csv"
        gnss_path = run_dir / "gnss.csv"
        index_path = run_dir / "frame_index.csv"

        t0 = None
        saved_frames = 0

        with sensors_path.open("w", newline="", encoding="utf-8") as sf, \
             gnss_path.open("w", newline="", encoding="utf-8") as gf, \
             index_path.open("w", newline="", encoding="utf-8") as xf:

            sensor_writer = csv.writer(sf)
            sensor_writer.writerow(SENSOR_COLUMNS)

            gnss_writer = csv.writer(gf)
            gnss_writer.writerow(["timestamp", "frame", "latitude", "longitude", "altitude"])

            index_writer = csv.writer(xf)
            index_writer.writerow(["frame", "timestamp", "path"])

            for tick in range(args.ticks):
                world.tick()

                imu = rig.next_imu()
                if imu is None:
                    break

                # Sim time relative to the start of recording, so sensors.csv
                # begins at 0.0 like the synthetic dataset does.
                if t0 is None:
                    t0 = imu.timestamp
                sim_time = imu.timestamp - t0

                # --- pothole logic -------------------------------------------
                wheels = wheel_positions(vehicle, config.WHEEL_POSITION_SCALE)
                for pothole, strike in tracker.update(sim_time, wheels):
                    # EITHER/OR, not both. Stacking them drove az to -25 when the
                    # unload's whole purpose is to hold it near ZERO -- the two
                    # mechanisms fight, and the recovery then skips the FSM's
                    # |az| < 2 window on its way back up. The unload alone
                    # cancels the suspension reaction, which is the physical
                    # model of a wheel falling into a hole.
                    if config.UNLOAD_ENABLED:
                        applier.schedule_unload(pothole.severity, strike)
                    else:
                        applier.schedule(pothole.severity, strike)
                applier.tick()

                # --- contract #1 row -----------------------------------------
                sensor_writer.writerow([
                    f"{sim_time:.6f}",
                    f"{imu.ax:.6f}", f"{imu.ay:.6f}", f"{imu.az:.6f}",
                    f"{imu.gx:.6f}", f"{imu.gy:.6f}", f"{imu.gz:.6f}",
                    f"{rt.speed_of(vehicle):.6f}",
                    tracker.label(sim_time),
                ])

                gnss = rig.next_gnss()
                if gnss is not None:
                    gnss_writer.writerow([
                        f"{sim_time:.6f}", gnss.frame,
                        f"{gnss.latitude:.9f}", f"{gnss.longitude:.9f}",
                        f"{gnss.altitude:.3f}",
                    ])

                # --- frames + the sync table that closes issue #18 ------------
                for image in rig.drain_images():
                    path = run_dir / "frames" / f"{image.frame:08d}.png"
                    image.save_to_disk(str(path))
                    index_writer.writerow([
                        image.frame,
                        f"{image.timestamp - t0:.6f}",
                        f"frames/{path.name}",
                    ])
                    saved_frames += 1

                vehicle.apply_control(follower.step())

                if follower.finished:
                    logger.info("Route complete at tick %d (%.1f sim-seconds).", tick, sim_time)
                    break

                if tick % 4000 == 0 and tick:
                    logger.info("tick %d | %.1fs sim | %d hits | %d frames",
                                tick, sim_time, len(tracker.hits), saved_frames)

        ph.write_ground_truth(run_dir / "ground_truth.json", placed, tracker,
                              args.town, args.seed)

        with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
            json.dump({
                "town": args.town,
                "seed": args.seed,
                "vehicle": config.VEHICLE_BLUEPRINT,
                "fixed_delta_seconds": config.FIXED_DELTA_SECONDS,
                "sample_rate_hz": 1.0 / config.FIXED_DELTA_SECONDS,
                "target_speed_kmh": config.TARGET_SPEED_KMH,
                "impulse_delta_v": config.IMPULSE_DELTA_V,
                "impulse_ticks": config.IMPULSE_TICKS,
                "unload_enabled": config.UNLOAD_ENABLED,
                "unload_ticks": config.UNLOAD_TICKS,
                "unload_force_scale": config.UNLOAD_FORCE_SCALE,
                "imu_noise_accel_stddev": config.IMU_NOISE_ACCEL_STDDEV,
                "imu_noise_gyro_stddev": config.IMU_NOISE_GYRO_STDDEV,
                "camera_enabled": config.CAMERA_ENABLED,
                "frames_saved": saved_frames,
                "level": "A (scripted impulse, no custom geometry)",
            }, f, indent=2)

        logger.info("Done. %d hits, %d frames -> %s", len(tracker.hits), saved_frames, run_dir)
        return 0

    finally:
        if rig is not None:
            rig.destroy()
        if vehicle is not None:
            vehicle.destroy()
        # Leaving the server synchronous makes it appear frozen to every other
        # client, including the CARLA window itself. Always restore.
        world.apply_settings(original_settings)
        logger.info("Restored original world settings.")


if __name__ == "__main__":
    sys.exit(main())
