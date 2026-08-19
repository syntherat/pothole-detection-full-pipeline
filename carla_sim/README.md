# carla_sim — CARLA testbed (Level A)

Drives a fixed route in CARLA, seeds a jerk at known pothole coordinates, and records
data in **exactly** the schema the existing pipeline already consumes.

> **Status:** scaffold written, **never executed.** CARLA was not installed on the machine
> where this was authored. Treat every runtime claim as unverified until `verify_setup.py`
> passes. Design rationale: [`../context/15-carla-testbed-plan.md`](../context/15-carla-testbed-plan.md).

## Why Level A

A pothole is negative space, and Unreal collides with surfaces — so a texture fools the camera
but the IMU feels nothing. Level B carves real geometry, which needs a CARLA source build.
**Level A skips that** by defining potholes as data and seeding the event with a physics impulse.

It does **not** script all three phases. It scripts only the drop:

1. A downward impulse at the wheel makes the body accelerate down.
2. The suspension unloads → the accelerometer's contact force falls toward zero. **That is a
   genuine freefall reading, from the physics engine.**
3. The spring re-compresses as the body falls back. **That is a genuine impact spike.**

So two of `PotholeDetector`'s three phases come out of vehicle dynamics, not out of a script.

## Setup

```bash
pip install -r carla_sim/requirements.txt
```

Start the CARLA simulator separately, then:

```bash
python carla_sim/verify_setup.py
```

**Run this first, every time you change CARLA version.** It measures four things and prints two
values to paste into `config.py`. The critical one is the IMU gravity convention: `PotholeDetector`
assumes `az ≈ +9.81` at rest, and if this build disagrees, nothing will ever trigger and it will
look like a logic bug for days.

## Record a run

```bash
python carla_sim/scenario/drive_and_record.py
```

IMU only, much faster:

```bash
python carla_sim/scenario/drive_and_record.py --no-camera --ticks 40000
```

## Output

```
out/run_<ts>/
├── sensors.csv       timestamp,ax,ay,az,gx,gy,gz,speed,label   ← contract #1, EXACTLY
├── gnss.csv          real lat/lng from sensor.other.gnss
├── frame_index.csv   frame → timestamp → path   ← this is what closes issue #18
├── frames/<frame>.png
├── ground_truth.json pothole positions + which were actually driven over
└── run_meta.json     config snapshot for reproducibility
```

`sensors.csv` is column-identical to `synthetic_pothole_dataset.csv` **on purpose**. The FSM, the
RandomForest, fusion and the connector all run against it unchanged — that is what makes this
cheap, so do not "improve" the schema (Rule 4).

## Files

| File | Role |
|---|---|
| `config.py` | Every tunable. Two values start as `None` and must be measured by `verify_setup.py` |
| `verify_setup.py` | **P0.** IMU gravity convention, wheel position units, impulse API, tick rate |
| `scenario/potholes.py` | Registry, placement along the route, wheel-over detection, ground-truth labels |
| `scenario/impulse.py` | The jerk model. Read the design note at the top before changing it |
| `scenario/route.py` | Deterministic route + a minimal waypoint follower |
| `scenario/sensors.py` | Sensor rig; IMU/GNSS every tick, camera at 20 Hz |
| `scenario/drive_and_record.py` | Main loop |

## Things that will bite you

- **`config.WHEEL_POSITION_SCALE` starts as `None`** and the recorder refuses to run until you set
  it. CARLA has documented `WheelPhysicsControl.position` as both metres and centimetres; guessing
  wrong means wheel-over detection silently never fires. `verify_setup.py` measures it.
- **`IMPULSE_DELTA_V` is a starting guess.** The right value drives `az` below the FSM's 6.81 drop
  threshold without launching the car. Tune it against real traces.
- **CARLA's IMU is noiseless by default.** `config.py` sets stddevs matching `generate_dataset.py`.
  Train on noiseless data and you build a classifier that solves a problem that does not exist.
- **Sync mode must be restored.** Both entry points restore world settings in a `finally` block —
  leaving the server synchronous makes it look frozen to every other client, including the CARLA
  window itself.
- **Level A does not validate vision.** There is no 3D pothole for the camera to see. It validates
  the sensor stage and the plumbing. Vision is Level B's job.

## Next

1. Run `verify_setup.py`, fill in the two config values.
2. Record a short run, plot `az` against `label`, and confirm the trace actually shows
   drop → freefall → impact.
3. Tune `IMPULSE_DELTA_V` until the FSM fires reliably.
4. Write `integration/carla_frame_provider.py` (P3) to replace the mock frame and GPS lookups.
5. Point the orchestrator at `sensors.csv` and run the real cascade.
