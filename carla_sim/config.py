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
# Raised from 20.0 to 120.0 on 2026-09-06 as headroom for client.load_world(),
# which is far slower than a plain connect.
#
# HONESTY NOTE: this was originally raised on a WRONG diagnosis. A load_world()
# call timed out and it was attributed to a slow map load; the simulator had in
# fact CRASHED ("LowLevelFatalError ... Shader compilation failures are Fatal")
# and the client was waiting on a dead process. A generous timeout is still the
# right default, but it fixed nothing that day. No real map-load duration has
# been measured yet -- if you need a justified number, measure one.
CARLA_TIMEOUT_S = 120.0
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
# RESTORED to 0.8 on 2026-09-06 after a tuning sweep showed NO value works.
# Do not "fix" issue #35 by changing this number -- it has been tried.
#
# Measured per 8,000 rows, Town03, tesla.model3, same seed:
#
#   delta_v   drop  freefall  impact   az_min   az_max
#   (none)      4       1        0      -7.1     18.8   <- road geometry alone
#   0.12       12       5        0      -8.5     18.8
#   0.35       12       1        0     -43.4     18.8
#   0.80       12       2        7    -111.3     21.1
#
# The FSM needs DROP -> FREEFALL(|az|<2) -> IMPACT(>19.81) on consecutive samples,
# resetting if az returns above G-0.5. Gentle enough to sit in the freefall window
# and the rebound never reaches the impact gate; strong enough to cross the impact
# gate and the drop skips the freefall window in a single sample. No value
# satisfies both, because CARLA's suspension answers an impulse with a 1-2 SAMPLE
# transient while the FSM expects the multi-sample shape of the hand-written
# synthetic pulses. See issue #35.
#
# Town03's own road geometry also produces 4 sub-threshold dips per 8,000 rows
# with NO potholes at all (verified with --potholes 0), reaching -7.1. Those are
# indistinguishable from a gentle impulse at the DROP gate.
IMPULSE_DELTA_V = 1.0       # m/s
IMPULSE_TICKS = 12            # spread over N ticks so it is a push, not a teleport

# --- wheel unloading (the freefall phase) ---
# Added 2026-09-06. An impulse gives the wheel a downward KICK, which is a 1-2
# sample transient. A real pothole UNLOADS the wheel for the entire time it is
# falling, holding the accelerometer near zero for tens of milliseconds. That
# sustained near-zero phase is precisely what PotholeDetector's min_air_time
# (0.01 s = 4 samples at 400 Hz) measures, and an impulse physically cannot
# produce it -- which is why every impulse-only setting failed. See issue #35.
#
# The model: hold a downward force roughly equal to the vehicle's weight at the
# striking wheel, cancelling the suspension's upward reaction so the body stops
# being held up. Release it and the spring re-compresses, producing the impact
# spike from real dynamics rather than a script.
UNLOAD_ENABLED = True
UNLOAD_TICKS = 19             # 47.5 ms of fall at 400 Hz. Must clear min_air_time
                              # (10 ms) AND give the recovery ramp long enough to
                              # climb back into the |az| < 2 window before release.
UNLOAD_FORCE_SCALE = 3.2      # MULTIPLES of vehicle weight held downward (not a
                              # fraction). 1.0 only cancels ~6.3 of the 9.81, so
                              # the plateau sat at ~3.5 and never entered the
                              # freefall window. Coupled to UNLOAD_TICKS -- see
                              # the sweep table in 21-configuration-and-tuning.md.

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
#
# MEASURED 2026-09-06, CARLA 0.9.16 packaged Windows build, Town10HD_Opt,
# vehicle.tesla.model3, on an RTX 4060 Laptop. Re-run verify_setup.py after any
# CARLA version change -- these are properties of the build, not of this repo.
WHEEL_POSITION_SCALE = 0.01   # 1.0 if wheel.position is metres, 0.01 if centimetres
                              # Raw wheel spread measured 300.93; the Model 3
                              # wheelbase is ~3.0 m, so the units are centimetres.
                              # (verify_setup prints ratio=62.80 against the
                              # bounding-box LENGTH 4.79 m, not the wheelbase --
                              # 3.01/4.79 = 0.63 is the expected ratio for a car,
                              # so that figure is consistent with cm, not evidence
                              # against it. Metres would imply a 300 m wheelbase.)
IMU_GRAVITY_AT_REST = 9.3483  # expected ~ +9.81 on az; anything else and the FSM
                              # thresholds in pothole_detection.py do not apply.
                              # Sign is POSITIVE, which is what matters -- the
                              # cascade would never trigger under a flipped or
                              # gravity-free convention.
                              #
                              # NOTE it is 9.35, not 9.81. The FSM gates are
                              # absolute (DROP < 6.81, IMPACT > 19.81), derived as
                              # 9.81 -/+ 3.0. Against a 9.35 rest they sit 2.54
                              # below and 10.46 above, so DROP is ~15% harder to
                              # reach than intended and IMPACT ~15% further away.
                              # Workable, but tune IMPULSE_DELTA_V against real
                              # traces rather than assuming the defaults transfer.
