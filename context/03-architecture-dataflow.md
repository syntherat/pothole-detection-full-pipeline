# 03 — Architecture & Data Flow

**Scope:** how the four subsystems connect; the exact end-to-end cascade with real numbers.
**Last verified against:** commit `3c586b8` (2026-08-19).

---

## Component diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        pothole_detect_physics/                                 │
│                                                                                │
│   synthetic_pothole_dataset.csv          pothole_ai_model.pkl                  │
│   80,000 rows @ 400 Hz, 200 s            RandomForest(200 trees, depth 12)     │
│              │                                       │                         │
│              ▼                                       │                         │
│   PotholeDetector (FSM)                              │                         │
│   IDLE→DROP→FREEFALL→IMPACT                          │                         │
│   → depth, length, air_time                          │                         │
└──────────────┬───────────────────────────────────────┬─────────────────────────┘
               │                                       │
               │        integration/sensor_adapter.py  │
               └──────────────► SensorSession.check_row(row) ◄──────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  physics_detected (bool)       │
                          │  physics_depth_estimate (m)    │
                          │  physics_length_estimate (m)   │
                          │  ai_score (0..1, predict_proba)│
                          └───────────────┬────────────────┘
                                          │
                    fusion.should_trigger_vision()
                    physics_detected AND ai_score >= 0.50
                                          │
                          ┌───────────────┴───────────────┐
                          │ no                            │ yes
                          ▼                               ▼
                    drop the row          integration/frame_provider.py  ⚠ MOCKED
                                          get_mock_frame(event_id)  → image path
                                          simulate_gps(timestamp)   → lat, lng
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        pothole_detection_app/                                  │
│                integration/vision_adapter.py → VisionSession.confirm()         │
│                                                                                │
│   TwoStageDetector.detect_potholes(frame, conf=0.35)                           │
│     stage A: road_seg.pt → visible-road mask (7-class, occluders removed)      │
│     stage B: best.pt      → pothole boxes                                      │
│     hard-filter: drop boxes whose centre is off-road                           │
│   → vision_score = mean(box confidences), or 0.0 if none                       │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                              fusion.fuse()
                    0.4 * ai_score + 0.6 * vision_score >= 0.50
                                     │
                                     ▼
                     integration/pave_connector.py
                     append to integration/pave_events.json
                                     │
                        polled every 3000 ms over HTTP
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          pothole_map_ui/                                       │
│   pollPaveEvents() → PotholeGuard.reportDetection(lat, lng, name, 'Both')      │
│   → addPothole() → red marker + live feed + badge + toast                      │
│   Geolocation watchPosition() → car marker → checkProximity()                  │
│   → Haversine < 100 m → proximity card pulses + toast                          │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## The cascade, step by step

Driven by `integration/orchestrator.py::run()`.

### Setup (once per run)
1. Read the whole CSV with pandas (`limit` truncates with `.head(limit)`).
2. `SensorSession()` — constructs a fresh `PotholeDetector` and `joblib.load`s the `.pkl`.
   **The detector is stateful.** One session per continuous drive; never share across vehicles or sessions.
3. `VisionSession()` — constructs the `TwoStageDetector`. YOLO models are stateless, so one instance serves the whole run.

### Per row
4. Build a `PotholeCandidateEvent` with a fresh `uuid4` and the row's `timestamp`.
5. **Stage 1** — `sensor.check_row(row)`:
   - `PotholeDetector.process_sample(...)` advances the FSM one sample and returns detection + estimates.
   - `ai_model.predict_proba(features)[0][1]` on the 7 feature columns gives `ai_score`.
   - Note: `predict_proba`, not `predict` — the cascade needs a graded score, not a hard 0/1.
6. **Gate** — `should_trigger_vision(physics_detected, ai_score)`: requires **both** `physics_detected` **and** `ai_score >= 0.50`. Neither alone promotes. Non-triggered rows are dropped with `continue`; no event record survives.
7. **Bridge (mocked)** — attach a frame path (`get_mock_frame`, deterministic per `event_id`) and a GPS fix (`simulate_gps`, a straight northward line from Bhopal at 0.000009 deg lat per second).
8. **Stage 2** — `vision.confirm(frame_path, conf=0.35)`: reads the image, runs the two-stage detector, sets `vision_score = mean(confidences)` (0.0 if no boxes) and `vision_confirmed = vision_score >= 0.35`.
9. **Fusion** — `fuse(ai_score, vision_score)`: `combined = 0.4*ai + 0.6*vision`; confirmed if `combined >= 0.50`.
10. Print a one-line trace, and if confirmed, `send_to_pave(event)` and keep the event.

### After the loop
11. Print the confirmed count and return the list of confirmed events.

---

## The three transports

### 1. Python-to-Python — direct calls through adapters
No IPC, no queue. The orchestrator holds live objects. `sys.path` manipulation in each adapter makes the sibling folders importable:

- `sensor_adapter.py` inserts `pothole_detect_physics/detector_py/` and imports `pothole_detection` by bare name.
- `vision_adapter.py` inserts the **repo root** and imports the dotted path `pothole_detection_app.app.two_stage_detection` (works via implicit namespace packages — there are no `__init__.py` files anywhere).

### 2. Python-to-browser — `pave_events.json`
Deliberately the dumbest thing that works.

- `pave_connector.send_to_pave()` reads the whole file, appends one record, rewrites it. Not concurrency-safe; a single-writer prototype.
- `app.js` fetches `../integration/pave_events.json` with `cache: 'no-store'` every 3 s, dedupes on `event_id` via a `Set`, and reports unseen events through `PotholeGuard`.
- **Requires HTTP.** On `file://` the fetch fails CORS and the console warns each tick.
- Designed to be swappable: replace `send_to_pave` with an HTTP POST and `pollPaveEvents` with a socket, and nothing else changes.

### 3. Anything-to-browser — `window.PotholeGuard`
The public JS API for injecting detections from any source:

```js
window.PotholeGuard.reportDetection(lat, lng, locationName, model);
window.PotholeGuard.getPotholes();
window.PotholeGuard.isDetected();
```

---

## Independent paths that bypass the cascade

Not everything flows through the orchestrator. Three other paths exist and are all legitimate:

| Path | Entry | Output | Reaches the map? |
|---|---|---|---|
| Vision-only, interactive | `run_app.py` → `PotholeAppFiltered` | Annotated frames + `events.jsonl` with **dummy GPS** near San Francisco | **Yes**, via `integration/filtered_gui_adapter.py` (since 2026-08-20) |
| Vision-only, enhanced GUI | `app/main_enhanced.py` | Annotated images/videos, CSV/JSON export | No |
| Vision-only, CLI | `scripts/predict_videos.py` | Annotated videos + CSV summary | No |
| Sensor-only, standalone | `Model/run_detector_on_dataset.py` | Console metrics + matplotlib plot | No |

**Bridged 2026-08-20:** `pothole_app_filtered.py` writes a rich per-event record (ID, GPS, timestamp, frame paths, detection count, and now a mean confidence) to `output/video_detect_*/metadata/events.jsonl`. That shape is deliberately not contract #7; `integration/filtered_gui_adapter.py` translates it and publishes through the same writer the cascade uses. Both mappings are in [`20-data-contracts.md`](20-data-contracts.md).

---

## Where the score comes from at each hop

| Hop | Value | Range | Meaning |
|---|---|---|---|
| Physics FSM | `physics_detected` | bool | The `DROP→FREEFALL→IMPACT` pattern completed with a plausible air time |
| RandomForest | `ai_score` | 0.0–1.0 | `predict_proba` for class 1 on a **single sample's** 7 features |
| Gate | `sensor_triggered` | bool | `physics_detected and ai_score >= 0.50` |
| YOLO | `vision_score` | 0.0–1.0 | Mean confidence across surviving boxes; 0.0 if none |
| Fusion | `final_confidence` | 0.0–1.0 | `0.4*ai_score + 0.6*vision_score` |
| Fusion | `final_decision` | bool | `final_confidence >= 0.50` |

**Consequence of the weights worth internalising:** with `vision_score = 0`, `ai_score` alone can never reach 0.5 (max 0.4), so **a confirmed event always requires the camera to see something**. With `ai_score = 1.0` (typical for a triggered row), a `vision_score` of just 0.167 clears the bar. In practice the sensor stage decides *when to look*, and the vision stage decides *whether it counts* — but only weakly. Tuning notes: [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md).

---

## Design principles to preserve

1. **Cheap gates expensive.** Never invert this. Running YOLO on every row defeats the architecture.
2. **Adapters own the translation.** Subsystem internals do not know about each other. See Rule 3.
3. **One shared event shape.** `PotholeCandidateEvent` accumulates state as it moves through stages; each stage fills its own fields and leaves the rest alone.
4. **Mocks are isolated.** All fakery lives in `frame_provider.py`, one file, clearly labelled. Real GPS and real frames land there and nowhere else.
5. **The transport is replaceable.** `pave_connector.py` is a single function with one job.
