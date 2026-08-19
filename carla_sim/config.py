"""
carla_sim/config.py

Every tunable for the Level A testbed in one place.

Level A = potholes defined as data, jerk seeded with a physics impulse.
No custom meshes, no CARLA source build. See context/15-carla-testbed-plan.md.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"

# --- connection ---
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT_S = 20.0
TOWN = "Town03"

# --- determinism ---
# Same seed + same route + same pothole placement => byte-comparable runs.
RANDOM_SEED = 42

# --- simulation rate ---
# 0.0025 s = 400 Hz, matching the synthetic dataset the FSM and RandomForest
# were fitted to, so neither needs retuning to get a first result.
#
# If wall-clock is too slow, 0.01 (100 Hz) is the fallback -- but then
# PotholeDetector.min_air_time (0.01 s) becomes ONE sample and stops being a
# meaningful gate. See context/21-configuration-and-tuning.md.
FIXED_DELTA_SECONDS = 0.0025

# PhysX substepping. CARLA requires:
#   FIXED_DELTA_SECONDS <= MAX_SUBSTEP_DELTA_TIME * MAX_SUBSTEPS
SUBSTEPPING = True
MAX_SUBSTEP_DELTA_TIME = 0.00125
MAX_SUBSTEPS = 10

# --- vehicle ---
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
TARGET_SPEED_KMH = 30.0

# --- IMU ---
# Mounted near the centre of the vehicle, roughly where a phone or a bolted-on
# IMU would sit. Height matters: the further from the roll axis, the more a
# one-sided pothole strike shows up in the lateral channels.
IMU_LOCATION = (0.0, 0.0, 1.0)  # x, y, z metres, vehicle frame

# CARLA's IMU defaults to ZERO noise, which is unrealistically clean -- a model
# trained on noiseless data solves a fake-easy problem. These match the
# distributions used by generate_dataset.py so the two datasets are comparable.
IMU_NOISE_ACCEL_STDDEV = 0.2   # m/s^2, per axis
IMU_NOISE_GYRO_STDDEV = 0.02   # rad/s, per axis

# --- camera ---
CAMERA_ENABLED = True
CAMERA_LOCATION = (1.6, 0.0, 1.4)   # dash-mounted, looking forward
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90.0
CAMERA_HZ = 20.0                    # NOT every tick -- 400 Hz imagery is neither
                                    # needed nor affordable

# --- potholes ---
NUM_POTHOLES = 25
POTHOLE_SPACING_M = 45.0     # along the route, so the FSM sees clean gaps
POTHOLE_RADIUS_M = 0.35      # >= a realistic tyre contact patch
POTHOLE_LATERAL_OFFSET_M = 0.75   # from lane centre, so one side hits at a time

# Severity 0-1 maps to impulse magnitude. Randomised per pothole to give the
# classifier a range of event strengths rather than 25 identical hits.
POTHOLE_SEVERITY_RANGE = (0.35, 1.0)

# --- impulse model (Level A) ---
# We seed ONLY the drop. The suspension unloading produces the freefall reading
# and the spring re-compressing produces the impact spike, so two of the FSM's
# three phases come out of the physics engine rather than a script.
#
# Expressed as a target downward delta-v so it scales across vehicle masses:
#   impulse [N.s] = mass [kg] * severity * IMPULSE_DELTA_V
#
# STARTING GUESS ONLY. Tune against real traces in P4 -- the right value is the
# one that reliably drives az below PotholeDetector's 6.81 drop threshold
# without launching the car.
IMPULSE_DELTA_V = 0.8        # m/s
IMPULSE_TICKS = 2            # spread over N ticks so it is a push, not a teleport

# --- ground truth ---
# label = 1 from the moment a wheel enters a pothole until this window expires.
# Needs to cover drop + freefall + impact; the synthetic dataset used 12 samples
# at 400 Hz = 0.03 s, so this is deliberately a little wider.
LABEL_WINDOW_S = 0.05

# --- run length ---
MAX_TICKS = 200_000          # hard stop; 200k ticks @ 400 Hz = 500 sim-seconds
ROUTE_LENGTH_M = 1200.0

# --- verified-in-P0 facts ---
# verify_setup.py measures these and prints the values to paste in here.
# Leave as None until measured -- code that depends on them fails loudly rather
# than silently assuming.
WHEEL_POSITION_SCALE = None   # 1.0 if wheel.position is metres, 0.01 if centimetres
IMU_GRAVITY_AT_REST = None    # expected ~ +9.81 on az; anything else and the FSM
                              # thresholds in pothole_detection.py do not apply
