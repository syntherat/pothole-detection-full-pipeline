# 10 — Physics / Sensor Pipeline

**Scope:** `pothole_detect_physics/` — the IMU state machine, the physics estimates, the synthetic dataset, and the RandomForest classifier.
**Last verified against:** commit `3c586b8` (2026-08-19).

---

## Mental model

A vehicle driving into a pothole produces a characteristic vertical-acceleration signature:

1. The wheel drops off the pothole's leading edge — `az` falls below the 1 g baseline.
2. The wheel (and briefly the body) is unsupported — `az` approaches 0, i.e. **freefall**.
3. The wheel strikes the trailing edge or the pothole floor — a large positive **impact** spike.

A **speed breaker** produces almost the mirror image: a rise, then a fall, and **no freefall phase**. That absence is what makes the two separable, and it is why the synthetic dataset deliberately injects speed breakers labelled `0`.

The subsystem attacks this two ways at once, and the integration layer requires both to agree:

- **Deterministic:** a finite state machine that recognises the pattern and derives physical quantities from it.
- **Statistical:** a RandomForest over the raw 7-channel sample, which gives a graded score.

---

## `detector_py/pothole_detection.py` — `PotholeDetector`

The core algorithm. ~150 lines, no dependencies beyond stdlib.

### Constructor parameters and derived thresholds

| Parameter | Default | Derived threshold | Meaning |
|---|---|---|---|
| `sampling_rate_hz` | 400.0 | — | Stored, currently not used in any computation |
| `drop_margin` | 3.0 | `drop_threshold = G - 3.0 = 6.81` | `az` below this starts a DROP |
| `impact_margin` | 10.0 | `impact_threshold = G + 10.0 = 19.81` | `az` above this from FREEFALL is an IMPACT |
| `freefall_threshold` | 2.0 | used directly | `abs(az)` below this is freefall |
| `min_air_time` | 0.01 s | used directly | Below this, reject as noise |
| `max_air_time` | 0.25 s | used directly | Above this, reject as implausible |

`G = 9.81` is a module constant.

### The state machine

```
        az < 6.81
IDLE ──────────────► DROP ──────────────► FREEFALL ──────────────► (finalize) ──► IDLE
                      │   abs(az) < 2.0      │    az > 19.81
                      │                      │
                      │ az > 9.31            │ elapsed > max_air_time
                      └──► reset to IDLE ◄───┘
```

- **IDLE:** waits for `az < drop_threshold`, records `t_drop_start`.
- **DROP:** if `abs(az) < freefall_threshold` → FREEFALL, record `t_freefall_start`. If `az > G - 0.5` (9.31) the signal recovered to baseline without a freefall — reset. This is the **speed-breaker rejection** in the FSM.
- **FREEFALL:** if `az > impact_threshold` → record `t_impact`, `speed_at_impact`, `impact_accel`, finalize, then reset. Otherwise **wait**, bounded by `max_air_time`.

#### The FREEFALL reset was changed on 2026-09-06 — read this before touching it

FREEFALL used to reset on **any** sample above `G - 0.5` that had not already cleared the impact gate:

```python
elif az > (G - 0.5):
    self.reset_state()
```

That requires the signal to cross from the freefall band to above 19.81 **in a single 2.5 ms sample**,
skipping the entire 9.31–19.81 range. `generate_dataset.py` writes precisely that discontinuity by hand —
`az[start+5:start+10] = freefall`, `az[start+10:start+12] = impact`, with nothing between, producing jumps
like `0.29 → 22.11`. So the detector worked on its own synthetic data and **could not work on any
continuously-varying signal**, because a real rebound passes *through* the band on its way up.

Measured: **33.3 %** of CARLA pothole samples land in that band, against **4.8 %** of synthetic ones.

The rule is now "wait, bounded by `max_air_time`" instead of "reset". A real CARLA strike recovers through
`… -1.24, 0.00, 0.39, 1.02` then rises `19.33 → 25.07`; the old rule discarded it at **19.33**, one sample
short.

**The event is still emitted on the threshold-crossing sample, not on the later peak.** Timestamping it at
the peak was tried and quietly halved cascade recall (80.7 % → 45.8 %): `pothole_detected` then lands on a
row that looks like ordinary driving, and the RandomForest filter downstream scores it 0 and discards the
event. The physics detector fired *more* while the cascade detected *less* — a failure that is invisible if
you only measure one stage.

**Effect on the synthetic dataset: none negative.** Event recall went 79.69 % → **80.73 %**, precision stays
100 %, spurious firings stay 0. The loosened rule recovers a couple of events that previously reset.

`process_sample()` is called **once per sample** and returns a dict every time; on non-event samples every field is `False`/`None`. `reset_state()` is public and clears everything.

### The physics

Computed in `_finalize_event()`:

```
t_air   = t_impact - t_freefall_start     # freefall duration
delta_t = t_impact - t_drop_start         # full drop-to-impact duration

depth  = G * t_air^2 / 8.0                # metres
length = speed_at_impact * delta_t        # metres
```

**Depth derivation.** Standard freefall gives `d = ½ g t²` for a full fall of duration `t`. The `/8` here is `½ g (t/2)²` — i.e. the model assumes the wheel falls for the **first half** of the measured air time and is decelerating/rising for the second half. This is a modelling assumption baked into the constant, not a typo. Changing the 8 changes every depth number the system has ever produced — treat it as a contract (Rule 4).

**Length derivation.** Distance travelled at the impact speed over the whole drop-to-impact window. Assumes constant speed across the event, which at 400 Hz over a few tens of milliseconds is reasonable.

**Air-time gate.** `t_air` outside `[0.01, 0.25]` returns a non-detection. At 400 Hz, 0.01 s is 4 samples — anything shorter is noise; 0.25 s would imply a ~0.077 m depth by the formula and an implausibly long airborne period for a road defect.

### Return shape

Always these five keys — see [`20-data-contracts.md`](20-data-contracts.md):

```python
{"pothole_detected": bool,
 "depth_estimate": float | None,
 "length_estimate": float | None,
 "air_time": float | None,
 "impact_acceleration": float | None}
```

The integration adapter consumes only the first three. `air_time` and `impact_acceleration` are currently dropped at the boundary — they are **not** in `PotholeCandidateEvent`. If you need them downstream, add fields to the schema (Rule 4).

### Statefulness — the most important operational fact

`PotholeDetector` carries FSM state between calls. Consequences:

- Samples **must** be fed in chronological order.
- One instance per continuous drive. `SensorSession` exists partly to enforce this; its docstring says so explicitly.
- Sharing an instance across two sessions can leave a half-open DROP that fuses the end of one drive to the start of another.
- If you ever parallelise row processing, you break this. Don't.

---

## `detector_py/generate_dataset.py` — the synthetic dataset

Generates `Data/synthetic_pothole_dataset.csv`. **Running it overwrites the committed 12.5 MB file and shifts every downstream number.** Rule 8 applies.

### Parameters

| Setting | Value |
|---|---|
| Sampling rate | 400 Hz (`dt = 0.0025 s`) |
| Duration | 200 s → 80,000 samples |
| Potholes injected | 200 |
| Speed breakers injected | 150 |

### Baseline signal

| Channel | Distribution |
|---|---|
| `ax`, `ay` | `N(0, 0.2)` |
| `az` | `N(9.81, 0.2)` |
| `gx`, `gy`, `gz` | `N(0, 0.02)` |
| `speed` | `U(10, 18)` — resampled **per row**, so it is white noise, not a realistic speed profile |

### Pothole injection — 12 samples (30 ms), `label = 1`

```
az[start   : start+5 ] = U(4, 7)     # drop
az[start+5 : start+10] = U(0.3, 1.0) # freefall
az[start+10: start+12] = U(18, 28)   # impact
+ N(0, 0.3) noise over all 12
```

### Speed breaker injection — 20 samples (50 ms), `label = 0`

```
az[start   : start+10] = U(12, 16)   # rise (above baseline)
az[start+10: start+20] = U(6, 8)     # fall (below baseline, but above freefall)
+ N(0, 0.3) noise
```

Deliberately labelled `0`. The fall never reaches the freefall band, so the FSM's DROP→reset branch rejects it and the classifier learns the same distinction.

### Interaction with the FSM — read this before tuning

The generated drop is `U(4, 7)`, but `drop_threshold` is `6.81`. So roughly the top 6 % of injected drops never trigger the IDLE→DROP transition on the drop samples themselves. Those events still usually get caught, because the *freefall* samples (0.3–1.0) are also below 6.81 and enter DROP on their own, then immediately satisfy the freefall condition on the next sample. The practical effect is a **shorter measured `t_air`** and therefore a **smaller depth estimate** for that subset. This is a known artefact of the synthetic generator, not of the algorithm.

Also note: events are placed at `np.random.randint(500, samples-500)` with **no collision check**, so injected events can overlap and corrupt each other's signatures.

### Output columns

`timestamp, ax, ay, az, gx, gy, gz, speed, label` — 80,000 rows plus header. Full spec in [`20-data-contracts.md`](20-data-contracts.md).

---

## `Model/features.py` — shared feature construction  *(added 2026-09-06)*

Training and evaluation must build **identical** features; when each script built its own, any divergence
would surface as an unexplained accuracy change rather than an error. Both now import from here.

| Function | Purpose |
|---|---|
| `ensure_event_ids(df)` | Guarantees `event_id` / `event_type`, deriving them from `label` runs when absent |
| `grouping_key(df)` | Group vector for `GroupShuffleSplit`. Background rows get a **unique** id each — one shared background group would force every negative onto one side of the split |
| `add_rolling_features(df)` | The item-4.3 context features. All windows are **causal** (`center=False`) — a centred window leaks the future into training |
| `full_feature_columns()` | Raw 7 + 14 rolling |

`ROLLING_WINDOWS = (8, 20, 40)` samples = 20/50/100 ms **at 400 Hz**. The source document specified those
counts at 1 kHz; the repository is 400 Hz throughout. Do not port the counts to another rate without
rescaling.

---

## `Model/train_ai_model.py` — the classifier

| Aspect | Value |
|---|---|
| Features | `ax, ay, az, gx, gy, gz, speed` — **7 columns, no `timestamp`** |
| Target | `label` |
| Split | 80/20, `random_state=42`, **stratified** |
| Model | `RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)` |
| Output | `Data/pothole_ai_model.pkl` via `joblib.dump`. **The pickle records its scikit-learn version** (currently 1.7.2), and `requirements.txt` pins to match — retraining means updating that pin |
| Reporting | `classification_report` + accuracy to stdout |

### What the model actually learns — and its limits

It is a **per-sample, memoryless** classifier. It sees one instant of a 7-channel vector and has no notion of the sequence. In practice this means it mostly learns "is `az` far from 9.81 in the way potholes are, and not in the way speed breakers are". It cannot represent the temporal pattern at all — that is entirely the FSM's job. The two are complementary by construction, which is exactly why the gate requires both.

Two consequences to keep in mind:

- **Class imbalance.** 200 potholes × 12 samples = 2,400 positive rows out of 80,000, i.e. **3 %**. Raw accuracy is a near-useless metric here; a model predicting all-zeros scores 97 %. Read precision/recall from the classification report, never the accuracy line.
- **Feature leakage risk is low but the domain gap is total.** The model has never seen a real accelerometer. Any accuracy figure from it describes the generator, not the road.

### Feature-column ordering is a contract

`FEATURE_COLUMNS` in `integration/sensor_adapter.py` must stay identical, and in the same order, to the `X` columns here. sklearn will warn or error on a mismatch. If you change one, change both (Rule 4).

---

## `Model/run_detector_on_dataset.py` — standalone demo

Runs the FSM sample-by-sample while using a single vectorised `ai_model.predict(X)` over the whole dataset up front (faster than per-row `predict`). Prints per-detection details, a detection summary, TP/FP/FN, and an "accuracy" figure, then draws a matplotlib plot of `az` with vertical lines at confirmed detections.

**Two caveats:**

1. **Import fixed 2026-08-19.** It lives in `Model/` while `PotholeDetector` lives in `detector_py/`, so it inserts `detector_py/` on `sys.path` before importing — the same pattern `integration/sensor_adapter.py` uses. It previously raised `ModuleNotFoundError` from any working directory.
2. Its "accuracy" is `TP / (TP + FP + FN)` — that is the **Jaccard index**, not accuracy. It is computed on the classifier's per-sample predictions across the whole dataset, and is unrelated to the FSM's detections despite appearing in the same summary block. Do not quote it as a system accuracy.

---

## Running it

See [`30-setup-and-run.md`](30-setup-and-run.md) for exact commands and the working-directory requirements. Dependencies are light — `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`. No torch, no GPU.

---

## Where to change what

| Goal | File | What to touch |
|---|---|---|
| Make the FSM more/less sensitive | `pothole_detection.py` | `drop_margin`, `impact_margin`, `freefall_threshold` |
| Reject more short/long events | `pothole_detection.py` | `min_air_time`, `max_air_time` |
| Change the depth model | `pothole_detection.py` | the `/8.0` in `_finalize_event` — **contract change** |
| Make the classifier stricter | `integration/fusion.py` | `SENSOR_THRESHOLD`, not the model |
| Change dataset realism | `generate_dataset.py` | injection blocks — **Rule 8** |
| Change model capacity | `train_ai_model.py` | `n_estimators`, `max_depth` — **Rule 8** |
