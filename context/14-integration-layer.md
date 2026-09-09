# 14 — Integration Layer

**Scope:** `integration/` — the cascade orchestrator, the adapters, fusion, the PAVE connector, and the test suite.
**Last verified against:** commit `3c586b8` (2026-08-19).

This is the newest and least battle-tested code in the repo. It is also the only place where all three subsystems meet.

---

## Design philosophy

Four rules the layer was built around. Preserve them.

1. **Nothing outside `integration/` knows about anything else outside `integration/`.** The physics and vision folders remain independently runnable projects. All translation happens in adapters here.
2. **One shared event shape.** `PotholeCandidateEvent` travels through the pipeline accumulating fields. No stage invents its own dict.
3. **All fakery in one file.** Every mock — frames, GPS — lives in `frame_provider.py`, labelled as such. When real synced data arrives, one file changes.
4. **The transport is trivially replaceable.** `pave_connector.send_to_pave()` is one function writing one file. Swap it for an HTTP POST and nothing upstream notices.

`integration/` has no `__init__.py` and is **not a package**. Modules import each other by bare name, which works because `python integration/orchestrator.py` puts `integration/` at `sys.path[0]`. `test_integration.py` inserts its own directory explicitly for the same reason.

---

## `schema.py` — `PotholeCandidateEvent`

A `@dataclass` with five field groups, filled progressively as the event moves through stages.

| Group | Fields | Filled by |
|---|---|---|
| identity | `event_id: str`, `timestamp: float` | orchestrator, at creation |
| location | `lat`, `lng` (`Optional[float]`) | `frame_provider.simulate_gps` — mocked |
| stage 1 | `physics_detected: bool`, `physics_depth_estimate`, `physics_length_estimate`, `ai_score`, `sensor_triggered: bool` | `sensor_adapter` + `fusion.should_trigger_vision` |
| stage 2 | `frame_path`, `vision_score`, `vision_confirmed` | `frame_provider` + `vision_adapter` |
| fusion | `final_decision`, `final_confidence` | `fusion.fuse` |

`as_dict()` returns a shallow copy of `__dict__`.

**Deliberate default asymmetry** — the tests enforce it: booleans that represent "has this happened yet" default to `False` (`physics_detected`, `sensor_triggered`), but `final_decision` defaults to `None` because "not yet computed" is genuinely different from "computed as false". Do not normalise this away.

Full field spec: [`20-data-contracts.md`](20-data-contracts.md).

---

## `sensor_adapter.py` — `SensorSession`

Wraps the physics FSM and the RandomForest behind one call.

```python
BASE_DIR / "pothole_detect_physics" / "detector_py"   # added to sys.path
from pothole_detection import PotholeDetector
AI_MODEL_PATH = BASE_DIR / "pothole_detect_physics" / "Data" / "pothole_ai_model.pkl"
FEATURE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "speed"]
```

`__init__` constructs a fresh `PotholeDetector()` and `joblib.load`s the model.

`check_row(row: pd.Series) -> dict` returns exactly four keys: `physics_detected`, `physics_depth_estimate`, `physics_length_estimate`, `ai_score`. (The test asserts this key set exactly — adding a key breaks it, intentionally.)

Two decisions worth understanding:

- **`predict_proba(features)[0][1]`, not `predict()`.** A cascade needs a graded score to threshold and to fuse. A hard 0/1 would collapse `fusion` into a boolean AND and throw away all the information the fusion weights exist to use.
- **The features are wrapped in a one-row `DataFrame`** rather than a raw array, so sklearn sees the same column names it was fitted with and does not warn.

**Statefulness — the class docstring says it and it matters:** `PotholeDetector` is a running FSM. One `SensorSession` per continuous drive/session. Never share one across vehicles or sessions; never feed rows out of order; never parallelise the row loop.

`FEATURE_COLUMNS` must match the training columns in `Model/train_ai_model.py` exactly, in order. Contract — Rule 4.

---

## `vision_adapter.py` — `VisionSession`

```python
sys.path.insert(0, str(BASE_DIR))
from pothole_detection_app.app.two_stage_detection import create_two_stage_detector
POTHOLE_MODEL_PATH = BASE_DIR / "pothole_detection_app" / "model" / "best.pt"
ROAD_MODEL_PATH    = BASE_DIR / "pothole_detection_app" / "model" / "road_seg.pt"
```

The road model is passed only `if ROAD_MODEL_PATH.exists()`, else `None`. **It is present**, so the vision stage runs the full two-stage path: detections outside the visible-road mask are dropped before `vision_score` is computed.

`confirm(frame_or_path, conf=0.35) -> dict`:

- Accepts a numpy BGR frame **or** a `str`/`Path`. Paths are read with `cv2.imread`, and a `None` result raises `ValueError` (a real failure, not a silent zero).
- Runs `detector.detect_potholes(frame, conf=conf)`.
- `vision_score = mean(box confidences)`, or `0.0` when there are no boxes.
- `vision_confirmed = vision_score >= conf`.

**Why the mean and not the max:** the mean penalises a frame where one strong detection sits among several weak ones, which is the shape of a false-positive-prone frame. The trade-off is that a single confident pothole among low-confidence noise scores lower than it should. If you change this, it changes every `final_confidence` in the system — Rule 4.

Note the class docstring: YOLO models are stateless, so **one `VisionSession` serves the whole run**. Only the sensor session is per-drive.

---

## `fusion.py` — the decision logic

Twenty lines, and the entire tuning surface of the cascade.

```python
SENSOR_THRESHOLD = 0.5     # ai_score needed, alongside physics_detected, to trigger vision
VISION_THRESHOLD = 0.35    # matches the YOLO GUI default
SENSOR_WEIGHT    = 0.4
VISION_WEIGHT    = 0.6
FUSION_THRESHOLD = 0.5

should_trigger_vision(physics_detected, ai_score) -> physics_detected and ai_score >= 0.5
fuse(ai_score, vision_score) -> (combined >= 0.5, combined)
    where combined = 0.4*ai_score + 0.6*vision_score
```

### What the numbers mean in practice

- **The gate is an AND.** Physics alone or AI alone never promotes. Two independent methods must agree before the expensive stage runs. `test_should_trigger_vision_requires_both_signals` pins this.
- **Vision outweighs sensor, 60/40.** With `vision_score = 0`, the maximum achievable `combined` is 0.4 — below the 0.5 threshold. **A confirmed event therefore always requires the camera to see something.** The sensor cannot confirm alone, by construction.
- **Conversely**, a triggered row typically has `ai_score` near 1.0, so `vision_score >= 0.167` clears the bar. The vision veto is real but weak.
- `VISION_THRESHOLD` (0.35) does two jobs: it is the YOLO confidence cutoff passed into inference, and the threshold for the `vision_confirmed` flag. `vision_confirmed` is recorded but **not** used in the final decision — `fuse` looks only at the raw score.

Tuning guidance and the effect of moving each number: [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md).

---

## `frame_provider.py` — the mocks ⚠

Everything fake in the pipeline lives here. The module docstring says so: *"Fakes the missing frame↔sensor link and GPS until real synced data exists."*

```python
MOCK_IMAGE_DIR = BASE_DIR / "pothole_detection_app" / "data" / "sample_images"
START_LAT, START_LNG = 23.2599, 77.4126        # Bhopal
METERS_PER_SECOND_LAT = 0.000009
```

**`get_mock_frame(event_id)`** — picks `pool[md5(event_id) % len(pool)]` from a **sorted** pool.
Deterministic per event id *and across processes* (MD5 rather than `hash()`, which Python salts per run;
sorted because `glob()` order is filesystem-dependent). Raises `FileNotFoundError` when no images exist.

**Image discovery.** `CANDIDATE_IMAGE_DIRS` is searched in order — `data/sample_images/`, then `input/`,
then the `dataset_v3` and `dataset_v2` test/val splits — and the first folder containing images wins, logged
once. So a user who has built a training dataset needs no configuration at all.

**`MockFrameProvider.get_frame()` returns `None`** when nothing is found, matching `CarlaFrameProvider`, and
warns once. The orchestrator then skips and counts those rows rather than dying — before 2026-08-20 the
raise killed the run on its first triggered event, which was the single most common failure.

The directory now exists and ships a README explaining what to put in it. Photos stay gitignored; the
README is tracked. **Supplying images is still on you** — they cannot be invented — but their absence is now
a reported skip rather than a crash.

**`simulate_gps(timestamp)`** — `lat = 23.2599 + timestamp * 0.000009`, `lng` constant. A straight northward line at roughly 1 m/s. Not a route, not a speed profile, and unrelated to the `speed` column in the CSV.

**When real data arrives**, this is the only file that changes: `get_mock_frame` becomes a lookup into a time-synchronised frame index, and `simulate_gps` becomes a read from the GPS log. Nothing else in the pipeline moves.

---

## `carla_frame_provider.py` — the real provider

`CarlaFrameProvider(run_dir)` — the thing `frame_provider.py` has been faking. Reads
`frame_index.csv` and `gnss.csv` from a run recorded by `carla_sim/scenario/drive_and_record.py`.

**The difference that matters:** the mock keys frames on `event_id`, which has no relationship to
anything. This keys on **time** — and a shared timebase is the only thing that makes a frame and a
sensor row belong together. That is what closes issue #18.

```python
get_frame(timestamp, event_id=None) -> str | None   # nearest frame within tolerance
get_gps(timestamp) -> tuple[float, float] | None     # linearly interpolated
coverage() -> dict                                   # diagnostics
```

- `DEFAULT_FRAME_TOLERANCE_S = 0.06`. The camera records at 20 Hz, so consecutive frames are 50 ms
  apart and worst-case nearest-frame error is 25 ms. The tolerance is deliberately **not** generous:
  pairing a sensor event with a frame from half a second later would silently reintroduce exactly
  the desynchronisation this class exists to remove.
- **Returning `None` is meaningful**, not a failure. It says the camera was not looking at that
  moment. Misses are counted in `misses_no_frame` / `misses_no_gps` so a run can be audited rather
  than silently losing events.
- Frame rows are sorted on load — `bisect` requires it and the CSV order is not assumed.
- Frames listed in the index but missing on disk are dropped with a warning.
- GPS is linearly interpolated between bracketing fixes. Safe here: a CARLA town spans a few hundred
  metres, so there is no antimeridian or pole to handle.

Self-check, no CARLA needed:
```
python integration/carla_frame_provider.py <run_dir>
```

`frame_provider.MockFrameProvider` is the matching object wrapper around the two mock functions, so
the two providers are interchangeable. The original `get_mock_frame` / `simulate_gps` functions are
untouched — `test_integration.py` calls them directly.

---

## `orchestrator.py` — the main loop

```python
DATASET_PATH = BASE_DIR / "pothole_detect_physics" / "Data" / "synthetic_pothole_dataset.csv"

run(limit: int | None = None,
    dataset_path: Path | str | None = None,   # any contract #1 CSV; defaults to synthetic
    frame_provider=None,                      # defaults to MockFrameProvider
    ) -> list[PotholeCandidateEvent]
```

CLI:
```
python integration/orchestrator.py --limit 2000
python integration/orchestrator.py --dataset <run>/sensors.csv --carla-run <run>
```

A sensor-triggered row whose provider returns no frame is **skipped and counted**
(`skipped_no_frame`, printed in the summary) rather than paired with an unrelated image.
A row whose provider returns no **GPS fix** is counted separately as `skipped_no_gps` and also
skipped: `CarlaFrameProvider.get_gps()` returns `None` outside the recorded GNSS window, and
publishing an event without a position would drop a marker at (0, 0).

> ⚠ **This signature was aspirational until 2026-09-06.** The block above describes what `run()`
> was designed to do; the code did not match it. `main()` passed `dataset_path` and
> `frame_provider` to a `run()` that accepted only `limit`, `argparse` was used but never imported,
> and `skipped_no_frame` was printed but never assigned — so the CLI raised `NameError` on the
> first line of `main()` and had evidently never been executed. Fixed by bringing the code up to
> this documentation rather than editing the documentation down to the code. See issue #31.

Loads the CSV (`dataset_path` or the synthetic default), optionally `.head(limit)`, constructs one `SensorSession` and one `VisionSession`, then per row: create the event, run Stage 1, apply the gate, `continue` if not triggered, ask the **provider** for a frame and a GPS fix (skipping and counting if either is unavailable), run Stage 2, fuse, print a trace line, and on confirmation call `send_to_pave` and collect the event.

Trace format:
```
[12.34s] sensor=0.98 vision=0.62 -> CONFIRMED (confidence=0.76)
```

Ends with `Total confirmed pothole events: N` and returns the list.

⚠ **`__main__` calls `run()` with no limit.** The file's own comment says to start small — but the call does not. 80,000 rows, with a YOLO forward pass on every triggered one. **Rule 8: do not run this unprompted.** Use `run(limit=2000)` from a Python shell, or edit the call, when experimenting.

Non-triggered rows are discarded entirely — no record survives. If you ever want recall analysis over rejected candidates, that is a change here.

---

## `carla_replay.py` — the CARLA vehicle feed  *(added 2026-09-06)*

Replays a recorded run onto the dashboard: walks `gnss.csv` in time order, publishes contract #12, and drops
pothole markers through `pave_connector.send_record_to_pave()` as they are detected.

```bash
python integration/carla_replay.py carla_sim/out/run_<ts> --speed 4 --reset --loop
```

**Markers are labelled `sensor (CARLA replay)`, never `hybrid (sensor+vision)`.** Level A has no hole in the
road mesh, so the dash camera records clean tarmac and the vision stage confirms nothing. Publishing
vision-confirmed markers from a Level A run would be a fabricated result.

**Why it does not call `SensorSession.check_row()` per row.** That method builds a one-row DataFrame and
calls `predict_proba` for every sample — about 25 minutes for a 40,000-row run, with a person waiting. The
classifier is stateless per row, so `carla_replay` scores the whole column in one vectorised call and drives
only the FSM sample by sample, because the FSM *is* stateful. It reuses the session's own
`physics_detector` and `ai_model` and the same `should_trigger_vision()` gate, so there is still one
definition of "detected". Total run time drops to about 11 seconds.

> ⚠ On the first recorded run this reports **0 detections** — see issue #35. The impulse overshoots the FSM's
> freefall window. The tool is working; the recording is not yet tunable input.

---

## `pave_connector.py` — the bridge

```python
EVENTS_FILE = Path(__file__).resolve().parent / "pave_events.json"

send_to_pave(event) -> None
```

Read the whole file (or start with `[]`), append one record, rewrite with `indent=2`.

Record shape (**not** the full event — a deliberately narrow projection):
```json
{"event_id": "...", "lat": 23.26, "lng": 77.41,
 "confidence": 0.76, "detected_by": "hybrid (sensor+vision)",
 "created_at": "2026-08-19T15:04:05.123456+00:00"}
```

`created_at` is timezone-aware UTC (`datetime.now(timezone.utc)`).

**Limitations, all acceptable for a prototype and all worth knowing:**
- Full read-modify-write per event. O(n²) over a run, and not concurrency-safe — a single writer only.
- The file grows without bound; nothing prunes it. It is gitignored, so deleting it is safe and resets the map's history.
- No error handling. A malformed file raises `JSONDecodeError` and kills the run.
- The map's dedupe means re-serving an existing file after a page refresh replays every event as new markers.

The module docstring names the intended upgrade path: a POST to a Flask endpoint, with nothing else changing.

---

## `test_integration.py`

Run with:
```
pytest integration/test_integration.py -v
```

Two deliberately separated classes of test, per the module docstring.

### A) Plumbing tests — valid today
| Test | Asserts |
|---|---|
| `test_schema_defaults` | Bools default `False`; `final_decision` defaults `None` |
| `test_fusion_logic_boundaries` | `fuse(1,1) → (True, 1.0)`; `fuse(0,0) → (False, 0.0)` |
| `test_should_trigger_vision_requires_both_signals` | The AND gate: neither signal alone triggers |
| `test_sensor_adapter_returns_expected_shape` | Exact key set; `physics_detected` is a bool; `0 <= ai_score <= 1`. Skipped if the `.pkl` is missing |
| `test_frame_provider_is_deterministic` | Same `event_id` → same frame. **Skips** cleanly when no sample images exist |
| `test_simulate_gps_moves_with_time` | Latitude advances, longitude does not |

### B) Stage-1 accuracy test — the one real number
`test_stage1_accuracy_against_real_labels` runs the sensor cascade over the first **2,000 rows** and compares `sensor_triggered` against the dataset's `label` column, printing precision and recall plus TP/FP/FN.

It **asserts nothing meaningful** (`precision >= 0.0`) on purpose — "good enough" depends on tuning goals. It exists as a **baseline to track while adjusting `SENSOR_THRESHOLD`**. Run it with `-s` to see the numbers.

**Scored per event since 2026-08-19.** `group_label_events()` groups contiguous `label==1` runs — one run
is one physical pothole — and a pothole counts as detected if the cascade triggered anywhere inside its
block or within `EVENT_MATCH_TOLERANCE_S` (0.02 s, converted to samples using the dataset's own timestep)
after it.

Measured on the first 2000 rows:

| Metric | Value |
|---|---|
| Events in window | 3 |
| **Event recall** | **1.00** |
| **Event precision** | **1.00** |
| Per-sample recall (printed for continuity) | 0.08 — ceiling is 1/12 = 0.083 |

The FSM fires on the 11th of each event's 12 labelled samples, exactly where `generate_dataset.py` injects
the impact block.

Before the fix this test scored per sample, which capped recall at 1/12 regardless of detector quality and
made the cascade look broken when it was catching everything. The per-sample line is still printed, marked
as not a quality signal, so older baselines remain comparable.

`test_label_event_grouping` covers the grouping logic without loading the model, and the accuracy test now
asserts that the analysed slice actually contains events — otherwise it would pass vacuously.

This remains the only accuracy measurement in the repo, it is still synthetic, and it still covers Stage 1
only. The vision half cannot be measured honestly until the mock frames are replaced with real synced ones.
er

| Goal | Where |
|---|---|
| Real camera frames | `frame_provider.get_mock_frame` |
| Real GPS | `frame_provider.simulate_gps` |
| Real transport to the UI | `pave_connector.send_record_to_pave` + `app.js::pollPaveEvents` |
| Add a third detection stage | New adapter module + new schema fields + a fusion term. Rules 3 and 4 |
| Carry depth/length to the map | Add to the `pave_connector` record, then to `app.js`. Rule 4 |
| Keep rejected candidates for analysis | Change the `continue` in `orchestrator.run` |
| Tune sensitivity | `fusion.py` only — never edit the subsystems for this |
