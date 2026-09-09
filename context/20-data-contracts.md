# 20 — Data Contracts

**Scope:** every data shape that crosses a module or subsystem boundary.
**Rule 4 applies to everything in this file: changing any of these means updating every consumer *and* this document, in the same change.**
**Last verified against:** commit `3c586b8` (2026-08-19).

---

## Contract inventory

| # | Contract | Producer | Consumer(s) |
|---|---|---|---|
| 1 | Sensor CSV columns | `generate_dataset.py` | `train_ai_model.py`, `sensor_adapter.py`, `orchestrator.py`, `run_detector_on_dataset.py`, tests |
| 2 | `PotholeDetector.process_sample()` return | `pothole_detection.py` | `sensor_adapter.py`, `run_detector_on_dataset.py` |
| 3 | RandomForest feature vector | `train_ai_model.py` | `sensor_adapter.py`, `run_detector_on_dataset.py` |
| 4 | `SensorSession.check_row()` return | `sensor_adapter.py` | `orchestrator.py`, tests |
| 5 | `VisionSession.confirm()` return | `vision_adapter.py` | `orchestrator.py` |
| 6 | `PotholeCandidateEvent` | `schema.py` | orchestrator, connector, tests |
| 7 | `pave_events.json` record | `pave_connector.py` | `app.js::pollPaveEvents` |
| 8 | `PotholeGuard` API | `app.js` | any external caller |
| 9 | Filtered-GUI event record | `pothole_app_filtered.py` | `integration/filtered_gui_adapter.py` |
| 10 | YOLO label formats | dataset prep scripts | Ultralytics |
| 11 | `TwoStageDetector` API | `two_stage_detection.py` | `vision_adapter.py`, both GUIs, `predict_videos.py` |

---

## 1. Sensor CSV — `Data/synthetic_pothole_dataset.csv`

80,000 data rows + header. 400 Hz over 200 s. 12.5 MB, committed.

| Column | Type | Unit | Notes |
|---|---|---|---|
| `timestamp` | float | s | `0.0` to `199.9975`, step `0.0025`. Monotonic — **order matters**, the FSM depends on it |
| `ax` | float | m/s² | Lateral. Baseline `N(0, 0.2)` |
| `ay` | float | m/s² | Longitudinal. Baseline `N(0, 0.2)` |
| `az` | float | m/s² | **Vertical — the signal that matters.** Baseline `N(9.81, 0.2)` |
| `gx` | float | rad/s | Baseline `N(0, 0.02)` |
| `gy` | float | rad/s | Baseline `N(0, 0.02)` |
| `gz` | float | rad/s | Baseline `N(0, 0.02)` |
| `speed` | float | m/s | `U(10, 18)` **resampled per row** — white noise, not a speed profile |
| `label` | float | 0.0/1.0 | 1.0 only for the 12 samples of an injected pothole. Speed breakers are 0.0 |

| `event_id` | int | — | **Added 2026-09-06.** `-1` for background; a distinct id per injected event. **Optional** |
| `event_type` | str | — | **Added 2026-09-06.** `none` / `pothole` / `speed_breaker`. **Optional** |

**Positive rate is ~3 %** (200 potholes × 12 samples / 80,000). Never judge a model on raw accuracy here.
That warning is now measured, not merely advised: the grouped-split fix moves accuracy by 0.11 points and
recall by 5.45 points on the same data. Accuracy is the metric least able to show the problem.

**The two new columns are additive and optional.** The committed CSV does not have them — it was generated
before they existed and, because the generator was unseeded until 2026-09-06, it cannot be reproduced. Any
consumer that needs them calls `Model/features.py::ensure_event_ids()`, which derives ids from runs of
consecutive `label == 1` when the columns are absent. That is the same event definition
`integration/test_integration.py::group_label_events()` uses.

This keeps `carla_sim/out/<run>/sensors.csv` **column-compatible**: it does not carry the new columns, and
nothing requires it to. Consumers select columns by name, so extra columns break no existing reader.

Any real-vehicle dataset must present **these exact column names** to work with the existing code unchanged.

---

## 2. `PotholeDetector.process_sample()` return

Called once per sample; returns a dict on **every** call, not only on detections.

```python
{
  "pothole_detected":     bool,           # True only on the sample that completes the pattern
  "depth_estimate":       float | None,   # metres, G*t_air^2/8
  "length_estimate":      float | None,   # metres, speed_at_impact * (t_impact - t_drop_start)
  "air_time":             float | None,   # seconds
  "impact_acceleration":  float | None,   # m/s², az at impact
}
```

All four optionals are `None` unless `pothole_detected` is `True`.

**Input signature:** `process_sample(timestamp, ax, ay, az, gx, gy, gz, speed)` — all keyword-passed by callers.

⚠ `air_time` and `impact_acceleration` are **dropped at the integration boundary** — `SensorSession.check_row` does not forward them and `PotholeCandidateEvent` has no fields for them. To use them downstream, extend both (Rule 4).

---

> **Behaviour change 2026-09-06 (shape unchanged).** The FSM no longer resets when a sample lands between
> rest and the impact threshold; it waits, bounded by `max_air_time`. The returned dict's keys and types are
> identical — but `pothole_detected` now fires on signals that previously never produced an event at all
> (any continuously-rising rebound). Consumers need no change; measured behaviour does differ. See
> `10-physics-sensor-pipeline.md` and issue #35.

---

## 3. RandomForest feature vector

Order is part of the contract. Defined in two places that must stay identical:

- `Model/train_ai_model.py` — the `X` column list
- `integration/sensor_adapter.py` — `FEATURE_COLUMNS`

```python
["ax", "ay", "az", "gx", "gy", "gz", "speed"]
```

**7 features. `timestamp` is excluded** (it would leak position in the synthetic sequence). `label` is the target.

Passed as a one-row `pd.DataFrame` so sklearn sees fitted column names. Output used: `predict_proba(X)[0][1]` — the probability of class 1.

---

## 4. `SensorSession.check_row()` return

```python
{
  "physics_detected":        bool,
  "physics_depth_estimate":  float | None,
  "physics_length_estimate": float | None,
  "ai_score":                float,        # 0.0-1.0
}
```

**Exactly these four keys.** `test_sensor_adapter_returns_expected_shape` asserts the key set with `==`, so adding a key fails the test — deliberately, to force a conscious contract change.

---

## 5. `VisionSession.confirm()` return

```python
{
  "vision_score":     float,   # mean confidence over surviving boxes, 0.0 if none
  "vision_confirmed": bool,    # vision_score >= conf
}
```

Input: a numpy BGR frame or a `str`/`Path`. An unreadable path raises `ValueError`.

---

## 6. `PotholeCandidateEvent` — `integration/schema.py`

The pipeline's shared state object. Fields are filled progressively.

```python
@dataclass
class PotholeCandidateEvent:
    # identity — required
    event_id: str                              # uuid4 hex string
    timestamp: float                           # seconds, from the sensor clock

    # location — mocked today
    lat: Optional[float] = None
    lng: Optional[float] = None

    # stage 1 — sensor
    physics_detected:        bool = False
    physics_depth_estimate:  Optional[float] = None
    physics_length_estimate: Optional[float] = None
    ai_score:                Optional[float] = None   # predict_proba, 0-1
    sensor_triggered:        bool = False             # physics AND ai_score >= threshold

    # stage 2 — vision
    frame_path:      Optional[str]   = None
    vision_score:    Optional[float] = None           # mean YOLO confidence, 0-1
    vision_confirmed: Optional[bool] = None

    # fusion
    final_decision:   Optional[bool]  = None
    final_confidence: Optional[float] = None

    def as_dict(self): ...                             # shallow copy of __dict__
```

**Default policy — enforced by `test_schema_defaults`:**
- `physics_detected` and `sensor_triggered` default to `False` — "has not happened".
- `final_decision` defaults to `None` — "not yet computed", which is genuinely distinct from "computed as false".

Do not normalise these to a single convention.

---

## 7. `pave_events.json` — the Python↔browser contract

The narrowest and most important contract in the repo: it is the only thing the map UI knows about the detection pipeline.

**File:** `integration/pave_events.json` — a JSON **array**, rewritten in full on each append, `indent=2`. Gitignored. Absent until the first confirmed event.

```json
[
  {
    "event_id":    "3f2b1c8a-...",              // uuid4 — the UI dedupes on this
    "lat":         23.259923,
    "lng":         77.4126,
    "confidence":  0.7612,                       // final_confidence, 0-1
    "detected_by": "hybrid (sensor+vision)",     // constant string
    "created_at":  "2026-08-19T15:04:05.123456+00:00"  // tz-aware UTC ISO 8601
  }
]
```

**Deliberately a projection, not the whole event.** Depth, length, `ai_score`, `vision_score` and `frame_path` are all withheld. Widening it means changing `pave_connector.py`, `app.js`, and this document together.

**Two producers, one writer.** Everything goes through `pave_connector.send_record_to_pave(record)`, so the shape is defined in exactly one place:

| Producer | Via | `detected_by` | `confidence` |
|---|---|---|---|
| Sensor+vision cascade | `send_to_pave(event)` | `"hybrid (sensor+vision)"` | `final_confidence` |
| Filtered GUI video runs | `integration/filtered_gui_adapter.py` | `"Image Model"` | mean box confidence, or `null` for runs recorded before 2026-08-20 |

⚠ The two `detected_by` vocabularies do not match: the cascade writes a free-form string while the adapter uses the map's own `MODELS` wording. The dashboard displays whichever string it gets, so this is cosmetic — but worth aligning if you touch either.

**Consumer behaviour** (`app.js::pollPaveEvents`):
- Polls every 3,000 ms with `cache: 'no-store'`.
- Skips any `event_id` already in `seenEventIds`.
- Maps to `PotholeGuard.reportDetection(lat, lng, label, model)` where `model` is `detected_by` (falling back to `'Both'`) and `label` appends the confidence when there is one.
- **`confidence` may be `null`.** Fixed 2026-08-20 — `app.js` previously called `.toFixed(2)` unconditionally, which threw on null and aborted the remaining events in that batch.
- **`detected_by` is now read.** It was previously ignored and every event was labelled `'Both'`, so camera-only detections claimed to be sensor+vision.
- A 404 is normal and silent.

---

## 12. `integration/vehicle_position.json` — the CARLA vehicle feed  *(added 2026-09-06)*

Written by `integration/carla_replay.py`, polled by `app.js` every 200 ms **in CARLA mode only**. Single
current position, overwritten in place — not a log.

| Key | Type | Meaning |
|---|---|---|
| `lat` / `lng` | float | Current fix, straight from the run's `gnss.csv`. Near (0, 0) for a CARLA town |
| `heading` | float | Compass bearing 0-360, computed from this fix to the next. Rotates the car icon |
| `timestamp` | float | Simulation seconds within the run |
| `progress` | float | 0.0-1.0 through the replay |
| `run` | str | Run directory name, e.g. `run_20260906_195639` |
| `updated_at` | str | ISO 8601 UTC |

Absent until a replay starts, and the poller stays silent about that — unlike a missing road network, which
is shouted about, because a blank map and a broken fetch look identical.

**This is what makes proximity alerts work in CARLA mode.** `checkProximity()` needs a vehicle position, and
browser geolocation cannot supply one: it reports wherever the browser is, which is thousands of km from a
town geo-referenced at (0, 0).

---

## 8. `window.PotholeGuard` — the JS injection API

```js
window.PotholeGuard.reportDetection(lat, lng, locationName = 'Road', model = 'Image Model') // → void
window.PotholeGuard.getPotholes()  // → Array<Pothole>
window.PotholeGuard.isDetected()   // → boolean
```

Internal `Pothole` object:
```js
{ id: "PH-001",           // UI-local sequential id, NOT the integration event_id
  lat, lng,
  locationName: string,
  detectedBy: string,     // conventionally 'Image Model' | 'Math Model' | 'Both'
  timestamp: Date }
```

**Two id namespaces coexist:** the integration UUID (used for dedupe) and the UI's `PH-NNN` (used for display). They never meet.

---

## 9. Filtered-GUI event record — `events.jsonl`

Written by `pothole_app_filtered.py` during realtime video, one JSON object per line, plus a per-event `<EVENT_ID>.json` and an `events_summary.json` array at the end of the run.

```json
{
  "id":              "PH-20260819153045-000142-0007",
  "potholes_detected": 3,
  "confidence":      0.8134,
  "latitude":        37.7751234,
  "longitude":      -122.4193456,
  "timestamp":       "2026-08-19T15:30:45.123456+00:00",
  "frame_index":     142,
  "frame_path":      "PH-20260819153045-000142-0007.jpg",
  "annotated_path":  "PH-20260819153045-000142-0007.jpg",
  "road_mask_path":  "PH-20260819153045-000142-0007.png"   // or null
}
```

Id format: `PH-<YYYYmmddHHMMSS>-<frame_idx:06d>-<saved_seq:04d>`.
Path fields are **basenames only**, relative to the sibling `frames/`, `annotated/`, `road_mask/` directories under `output/video_detect_<ts>/`.

⚠ **GPS here is fake** — jittered around a randomised base near San Francisco. Note it differs from the integration mock (Bhopal).

`confidence` is the **mean over the boxes that survived class and road-mask filtering**, matching how `integration/vision_adapter.py` scores a frame. Added 2026-08-20; runs recorded before that have no such key, and consumers must treat it as optional.

This shape is deliberately **not** contract #7 — the GUI keeps its own richer record (frame paths, frame index, detection count) and `integration/filtered_gui_adapter.py` translates. Mapping:

| events.jsonl | pave_events.json |
|---|---|
| `id` | `event_id` |
| `latitude` / `longitude` | `lat` / `lng` |
| `confidence` | `confidence` |
| `timestamp` | `created_at` |
| — | `detected_by` = `"Image Model"` |
| `potholes_detected` | dropped |

Records without coordinates are **dropped**, not placed with a guessed position.

---

## 10. YOLO label formats

**Detection** — `data/dataset_v*/{split}/labels/<image>.txt`, one line per object, all values normalised 0–1:
```
0 0.425 0.512 0.156 0.178
class_id center_x center_y width height
```
Single class: `0 = pothole`.

**Segmentation** — polygon vertices, normalised:
```
0 x1 y1 x2 y2 x3 y3 ...
```
Class list depends on the preset — `core`: `visible_road, vehicle, pedestrian, shadow`; `extended` adds `vegetation, roadside_object, road_obstacle`.

**The shipped `model/road_seg.pt` uses the `extended` preset** — this ordering is a contract between the
weights and every keyword-matching code path:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| class | `visible_road` | `vehicle` | `pedestrian` | `shadow` | `vegetation` | `roadside_object` | `road_obstacle` |

Both road-mask implementations resolve these **by substring match on the name**, not by index, so
renaming a class silently changes which pixels count as road. See the resolution trace in
[`11-vision-pipeline.md`](11-vision-pipeline.md). Retraining with a different preset, or reordering the
`names` list, is a contract change (Rule 4).

**VOC XML input** — `<annotation><size>` plus `<object><name>` and `<bndbox>` with absolute `xmin/ymin/xmax/ymax`. Converted by `merge_datasets.voc_to_yolo()` and `prepare.voc_xml_to_yolo_lines()`.

---

## 11. `TwoStageDetector` API

The most widely reused API in the repo — imported by the integration layer, both GUIs, and the video CLI. Changes ripple everywhere.

```python
create_two_stage_detector(pothole_model_path, road_model_path=None) -> TwoStageDetector

detector.use_road_seg  # bool — False when no road model loaded

detector.get_road_mask(frame, lowres_width=None) -> np.uint8 mask
    # 255 = road, 0 = not road; all-255 when use_road_seg is False

detector.detect_potholes(frame, conf=0.35, return_mask=False,
                         road_mask=None, lowres_width=None)
    -> Results | (Results, mask)
    # results.boxes is ALREADY filtered to on-road detections

detector.visualize(frame, results, road_mask=None, show_mask=True) -> np.ndarray
```

**The critical guarantee:** `results.boxes` returned by `detect_potholes` is the *filtered* set. `len(results.boxes)` and any confidence aggregate over it already exclude off-road detections. `vision_adapter.confirm()` depends on this.

Mask convention: `255` road / `0` not road, tested with `> 127` downstream.
