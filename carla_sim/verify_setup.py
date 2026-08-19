"""
carla_sim/verify_setup.py

P0 -- RUN THIS BEFORE WRITING OR TRUSTING ANYTHING ELSE.

Measures the four facts the rest of the testbed assumes. Each one is cheap to
check and expensive to get wrong:

  1. IMU gravity convention   -- the whole FSM assumes az ~= +9.81 at rest.
                                 If CARLA reports gravity-free values or a
                                 flipped Z axis, NOTHING will ever trigger and
                                 it will present as a logic bug for days.
  2. Wheel position units     -- documented inconsistently as metres or
                                 centimetres. Measured here, not assumed.
  3. Impulse API shape        -- which of the add_impulse* methods this build
                                 actually exposes.
  4. Achievable tick rate     -- whether 400 Hz is practical on this machine.

Usage:
    python carla_sim/verify_setup.py

Paste the printed values into config.py.
"""

from __future__ import annotations

import math
import statistics
import sys
import time

try:
    import carla
except ImportError:
    sys.exit(
        "carla module not found.\n"
        "  pip install carla\n"
        "and make sure the CARLA simulator itself is running."
    )

import config


def _banner(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def check_imu_gravity(world, vehicle) -> float | None:
    """
    Park the vehicle and read the accelerometer.

    PotholeDetector treats az as specific force with gravity included:
        rest      az ~= 9.81
        DROP      az  <  6.81
        IMPACT    az  > 19.81
    """
    _banner("1. IMU GRAVITY CONVENTION")

    bp = world.get_blueprint_library().find("sensor.other.imu")
    # Noise off for this measurement -- we want the raw convention, not a sample
    # from a noisy distribution.
    for attr in ("noise_accel_stddev_x", "noise_accel_stddev_y", "noise_accel_stddev_z"):
        if bp.has_attribute(attr):
            bp.set_attribute(attr, "0.0")

    imu = world.spawn_actor(
        bp,
        carla.Transform(carla.Location(*config.IMU_LOCATION)),
        attach_to=vehicle,
    )

    samples: list[tuple[float, float, float]] = []
    imu.listen(lambda d: samples.append((d.accelerometer.x, d.accelerometer.y, d.accelerometer.z)))

    # Handbrake on, let the suspension settle before sampling.
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    for _ in range(400):
        world.tick()

    settled = samples[-200:]
    imu.stop()
    imu.destroy()

    if not settled:
        print("  FAIL: no IMU samples received.")
        return None

    ax = statistics.mean(s[0] for s in settled)
    ay = statistics.mean(s[1] for s in settled)
    az = statistics.mean(s[2] for s in settled)

    print(f"  At rest:  ax={ax:+.4f}  ay={ay:+.4f}  az={az:+.4f}  (m/s^2)")
    print(f"  Magnitude: {math.sqrt(ax*ax + ay*ay + az*az):.4f}")

    if abs(az - 9.81) < 0.5:
        print("\n  OK -- az ~= +9.81. Matches PotholeDetector's assumption.")
        print("  The FSM thresholds (6.81 / 19.81) apply as written.")
    elif abs(az + 9.81) < 0.5:
        print("\n  *** Z AXIS IS INVERTED (az ~= -9.81). ***")
        print("  Negate az in the recorder, or the FSM will never leave IDLE.")
    elif abs(az) < 0.5:
        print("\n  *** GRAVITY IS NOT INCLUDED (az ~= 0 at rest). ***")
        print("  Add +9.81 to az in the recorder, or rewrite the FSM thresholds")
        print("  around a 0-baseline. The first option is far less invasive.")
    else:
        print(f"\n  *** UNEXPECTED BASELINE ({az:+.4f}). ***")
        print("  Do not proceed until you understand where this value comes from.")

    print(f"\n  -> set IMU_GRAVITY_AT_REST = {az:.4f} in config.py")
    return az


def check_wheel_units(vehicle) -> float | None:
    """
    Decide whether WheelPhysicsControl.position is metres or centimetres by
    comparing wheel spread against the vehicle's own bounding box.
    """
    _banner("2. WHEEL POSITION UNITS")

    physics = vehicle.get_physics_control()
    wheels = physics.wheels
    if not wheels:
        print("  FAIL: no wheels reported.")
        return None

    loc = vehicle.get_transform().location
    extent = vehicle.bounding_box.extent  # metres, half-extents

    print(f"  Vehicle location (m): x={loc.x:.2f} y={loc.y:.2f} z={loc.z:.2f}")
    print(f"  Vehicle half-extent (m): x={extent.x:.2f} y={extent.y:.2f}")
    for i, w in enumerate(wheels):
        print(f"  wheel[{i}].position: x={w.position.x:.3f} y={w.position.y:.3f} z={w.position.z:.3f}")

    # Wheels sit within a couple of metres of the vehicle origin. If the raw
    # numbers are ~100x that separation, they are centimetres.
    raw_spread = max(abs(w.position.x - wheels[0].position.x) for w in wheels)
    expected_spread = 2.0 * extent.x

    if expected_spread <= 0:
        print("  FAIL: degenerate bounding box.")
        return None

    ratio = raw_spread / expected_spread
    print(f"\n  wheelbase-ish spread: raw={raw_spread:.3f}  expected~={expected_spread:.3f}  ratio={ratio:.2f}")

    if 0.5 < ratio < 2.0:
        scale = 1.0
        print("  -> positions are in METRES. WHEEL_POSITION_SCALE = 1.0")
    elif 50.0 < ratio < 200.0:
        scale = 0.01
        print("  -> positions are in CENTIMETRES. WHEEL_POSITION_SCALE = 0.01")
    else:
        print("  *** INCONCLUSIVE. Inspect the numbers above by hand. ***")
        return None

    print(f"\n  -> set WHEEL_POSITION_SCALE = {scale} in config.py")
    return scale


def check_impulse_api(vehicle) -> None:
    """
    Report which impulse methods exist. impulse.py adapts to this at runtime,
    but knowing which path is taken matters when reading traces.
    """
    _banner("3. IMPULSE API")

    for name in ("add_impulse", "add_impulse_at_location", "add_angular_impulse", "add_force", "add_torque"):
        print(f"  {'OK  ' if hasattr(vehicle, name) else 'MISSING'}  {name}")

    if hasattr(vehicle, "add_impulse_at_location"):
        print("\n  -> off-centre impulse applied directly. Best case: roll and")
        print("     pitch come out of the engine.")
    else:
        print("\n  -> will fall back to add_impulse + add_angular_impulse with a")
        print("     manually computed r x J. Physically equivalent; see impulse.py.")


def check_tick_rate(world) -> None:
    """Measure real wall-clock throughput at the configured timestep."""
    _banner("4. TICK RATE")

    n = 400
    start = time.perf_counter()
    for _ in range(n):
        world.tick()
    elapsed = time.perf_counter() - start

    sim_seconds = n * config.FIXED_DELTA_SECONDS
    ratio = sim_seconds / elapsed

    print(f"  timestep:      {config.FIXED_DELTA_SECONDS} s ({1/config.FIXED_DELTA_SECONDS:.0f} Hz)")
    print(f"  {n} ticks in:  {elapsed:.2f} s wall clock")
    print(f"  sim time:      {sim_seconds:.2f} s")
    print(f"  real-time factor: {ratio:.3f}x")

    if ratio >= 1.0:
        print("\n  OK -- faster than real time.")
    elif ratio >= 0.1:
        minutes = (config.ROUTE_LENGTH_M / (config.TARGET_SPEED_KMH / 3.6)) / ratio / 60
        print(f"\n  Slower than real time. A {config.ROUTE_LENGTH_M:.0f} m route would take")
        print(f"  roughly {minutes:.0f} minutes of wall clock. Workable but slow.")
    else:
        print("\n  *** VERY SLOW. Consider FIXED_DELTA_SECONDS = 0.01 (100 Hz) and")
        print("      raising PotholeDetector.min_air_time to match. ***")


def main() -> None:
    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT_S)
    world = client.get_world()

    original = world.get_settings()
    vehicle = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
        settings.substepping = config.SUBSTEPPING
        settings.max_substep_delta_time = config.MAX_SUBSTEP_DELTA_TIME
        settings.max_substeps = config.MAX_SUBSTEPS
        world.apply_settings(settings)

        print(f"Connected. Map: {world.get_map().name}")
        print(f"CARLA client version: {client.get_client_version()}")
        print(f"CARLA server version: {client.get_server_version()}")

        bp = world.get_blueprint_library().find(config.VEHICLE_BLUEPRINT)
        spawn = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(bp, spawn)
        for _ in range(20):
            world.tick()

        check_imu_gravity(world, vehicle)
        check_wheel_units(vehicle)
        check_impulse_api(vehicle)
        check_tick_rate(world)

        _banner("DONE")
        print("Copy the two '-> set ...' values into carla_sim/config.py before")
        print("running drive_and_record.py.")

    finally:
        if vehicle is not None:
            vehicle.destroy()
        # Leaving the server in synchronous mode makes it look frozen to every
        # other client. Always restore.
        world.apply_settings(original)
        print("\nRestored original world settings.")


if __name__ == "__main__":
    main()
