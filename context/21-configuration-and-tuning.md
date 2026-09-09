# 21 — Configuration & Tuning

**Scope:** every threshold, constant and tunable in the system, where it lives, and what moving it does.
**Last verified against:** commit `3c586b8` (2026-08-19).

There is **no central config file.** Constants live at the top of the module that owns them. This table is the index.

---

## The tuning surface at a glance

| Constant | Value | File | Governs |
|---|---|---|---|
| `SENSOR_THRESHOLD` | 0.50 | `integration/fusion.py` | When the sensor stage promotes to vision |
| `VISION_THRESHOLD` | 0.35 | `integration/fusion.py` | YOLO conf cutoff + `vision_confirmed` flag |
| `SENSOR_WEIGHT` | 0.40 | `integration/fusion.py` | Sensor share of the fused score |
| `VISION_WEIGHT` | 0.60 | `integration/fusion.py` | Vision share of the fused score |
| `FUSION_THRESHOLD` | 0.50 | `integration/fusion.py` | Final confirm/reject cut |
| `drop_margin` | 3.0 | `pothole_detection.py` | DROP entry: `az < 9.81 - 3.0` |
| `impact_margin` | 10.0 | `pothole_detection.py` | IMPACT: `az > 9.81 + 10.0` |
| `freefall_threshold` | 2.0 | `pothole_detection.py` | FREEFALL: `abs(az) < 2.0` |
| `min_air_time` | 0.01 s | `pothole_detection.py` | Reject shorter events |
| `max_air_time` | 0.25 s | `pothole_detection.py` | Reject longer events |
| depth divisor | 8.0 | `pothole_detection.py` | `depth = G*t_air²/8` |
| `n_estimators` | 200 | `train_ai_model.py` | RandomForest trees |
| `max_depth` | 12 | `train_ai_model.py` | RandomForest depth |
| `_conf` | 0.35 | `app/utils.py` | Global YOLO confidence (enhanced GUI path). Read it with `get_conf_threshold()`, never by importing the name |
| `conf_threshold` | 0.35 | `pothole_app_filtered.py` | YOLO confidence (filtered GUI) |
| road seg `conf` | 0.50 | `pothole_app_filtered.py` | Segmentation confidence |
| `ROAD_MASK_STRIDE` | 10 | `main_enhanced.py`, `predict_videos.py` | Recompute the road mask every N frames |
| `ROAD_MASK_WIDTH` | 800 | `main_enhanced.py`, `predict_videos.py` | Downscale width for segmentation |
| `PROXIMITY_RADIUS_M` | 100 | `pothole_map_ui/app.js` | Driver alert radius |
| `PAVE_POLL_INTERVAL_MS` | 3000 | `pothole_map_ui/app.js` | Event poll cadence |
| `MAP_CENTER` | 23.2599, 77.4126 | `pothole_map_ui/app.js` | Fallback map centre (Bhopal) |
| `START_LAT/LNG` | 23.2599, 77.4126 | `frame_provider.py` | Mock GPS origin |
| `METERS_PER_SECOND_LAT` | 0.000009 | `frame_provider.py` | Mock GPS drift rate |
| `demo_base_lat/lon` | 37.7749, -122.4194 | `pothole_app_filtered.py` | Dummy GPS base (San Francisco) |
| `sampling_rate` | 400 Hz | `generate_dataset.py` | Dataset rate |
| `total_time` | 200 s | `generate_dataset.py` | Dataset length |
| `num_potholes` | 200 | `generate_dataset.py` | Injected positives |
| `num_speedbreakers` | 150 | `generate_dataset.py` | Injected hard negatives |
| detection-status revert | 4000 ms | `app.js` | TRUE→FALSE timer |
| toast duration | 3500 ms | `app.js` | Notification lifetime |
| live feed cap | 10 | `app.js` | Sidebar list length |

---

## CARLA testbed — `carla_sim/config.py`

**The two P0 values are no longer `None`.** Measured 2026-09-06 by `verify_setup.py` against CARLA 0.9.16
(packaged Windows build, Town10HD_Opt, `vehicle.tesla.model3`, RTX 4060 Laptop). They are properties of the
CARLA build, not of this repo — **re-run `verify_setup.py` after any CARLA version change.**

| Constant | Measured | Note |
|---|---|---|
| `IMU_GRAVITY_AT_REST` | **9.3483** | Sign is **positive**, which is the fact that mattered — a flipped or gravity-free convention would mean nothing ever triggers |
| `WHEEL_POSITION_SCALE` | **0.01** | Wheel positions are in centimetres |
| `CARLA_TIMEOUT_S` | 20.0 → **120.0** | Headroom for `load_world()`. See the honesty note in the file: it was raised on a wrong diagnosis |

### Wheel unloading — the freefall phase  *(added 2026-09-06)*

| Constant | Value | Meaning |
|---|---|---|
| `UNLOAD_ENABLED` | `True` | Use the sustained-unload model instead of the impulse |
| `UNLOAD_TICKS` | **19** | ~47 ms of fall at 400 Hz. Must exceed `min_air_time` (10 ms) with margin |
| `UNLOAD_FORCE_SCALE` | **3.2** | Multiples of vehicle weight held downward at the striking wheel |

**These two are coupled, and tuning one alone fails.** Force sets how deep the fall goes and therefore how
hard the landing is; duration sets how long the recovery ramp has to climb back into the FSM's ±2 freefall
window before the force is released. Too much force for the duration and the recovery is still at −2.1 when
the impact fires, so the freefall window is skipped. Too little and the rebound never clears 19.81. The
working pair was found by sweep:

| scale | ticks | freefall window entered? | peak `az` | detections |
|---|---|---|---|---|
| 1.0 | 8 | no (plateau ~3.5) | 12.4 | 0 |
| 1.6 | 8 | **yes**, 8 samples | 13.8 | 0 |
| 2.2 | 14 | **yes**, 7 samples | 18.1 | 0 |
| 3.2 | 14 | no (−2.13 → 16.34) | 21.8 | 0 |
| **3.2** | **19** | **yes**, 6 samples | **25.1** | **fires** |

`UNLOAD_ENABLED` and the impulse are **mutually exclusive** — applying both drove `az` to −25 when the
unload's entire purpose is to hold it near zero, and the two mechanisms fought.

---

**`IMU_GRAVITY_AT_REST` is 9.35, not 9.81, and the FSM gates are absolute.** `pothole_detection.py` uses
`DROP < 6.81` and `IMPACT > 19.81`, derived as `9.81 ∓ 3.0`. Against a 9.35 rest those sit 2.54 below and
10.46 above, so **DROP is about 15 % harder to reach than intended** and IMPACT about 15 % further away.
Workable, but it means `IMPULSE_DELTA_V` must be tuned against real traces rather than assumed to transfer.
This is the sort of quiet offset that presents as "the FSM never fires" much later.

`add_impulse_at_location` is **absent** on this build. `impulse.py` already detects that with `hasattr` and
falls back to `add_impulse` + `add_angular_impulse` with a manually computed `r × J`. No action needed.

Measured tick rate: 400 Hz runs at **0.235× real time** on this machine — roughly 10 minutes of wall clock
per 1200 m route. Relevant to open decision #2 (400 Hz vs 100 Hz).

---

## Cascade tuning — `integration/fusion.py`

This is where you tune the system as a system. **Prefer changing these over changing any subsystem's internals** (Rule 3).

### `SENSOR_THRESHOLD` (0.5) — how often the camera is consulted

The gate is `physics_detected AND ai_score >= SENSOR_THRESHOLD`.

| Direction | Effect |
|---|---|
| Lower (e.g. 0.3) | More rows promoted → more YOLO calls → higher recall, higher compute cost, more chances for the vision stage to reject |
| Raise (e.g. 0.7) | Fewer promotions → cheaper, but a missed promotion is unrecoverable — the event is gone |

**This is the recall ceiling of the whole system.** Nothing downstream can recover an event the gate dropped. Measure the effect with `test_stage1_accuracy_against_real_labels`, which exists precisely for this.

### `SENSOR_WEIGHT` / `VISION_WEIGHT` (0.4 / 0.6) — who decides

They sum to 1.0. Keep it that way, or `FUSION_THRESHOLD` stops meaning what it appears to mean.

Understand the current geometry before touching them:

- With `vision_score = 0`, max `combined` is **0.4 < 0.5**. **The camera has an absolute veto.** No amount of sensor confidence confirms an event alone.
- With `ai_score = 1.0` (typical for a promoted row), `vision_score >= 0.167` clears the bar. The veto is real, but easily satisfied.
- If you set `SENSOR_WEIGHT >= FUSION_THRESHOLD`, you **destroy the veto** — a confident sensor reading alone confirms, and the vision stage becomes decorative.

> ⚠️ **This relationship is now recited in a patent claim (2026-09-09).** The disclosure states that the
> weights and threshold are chosen such that the vision contribution is *necessary* for confirmation —
> which holds exactly while `SENSOR_WEIGHT < FUSION_THRESHOLD`. Setting them equal or inverted does not
> merely weaken the design; it makes the filed claim stop reading on the system. The *values* are free
> to move (the claim is worded as a property, not as 0.4/0.6/0.5) — the *inequality* is not. Agreed
> with the fusion owner before filing. See [`04-current-state.md`](04-current-state.md).

### `FUSION_THRESHOLD` (0.5) — the final cut

Raise for precision, lower for recall. Because vision dominates the blend, this mostly acts as a vision-confidence cut in disguise.

### `VISION_THRESHOLD` (0.35)

Two jobs: the `conf` passed into YOLO inference, and the cut for the `vision_confirmed` flag. Note that **`vision_confirmed` does not participate in `fuse()`** — the final decision uses the raw `vision_score`. Raising this raises the floor on which boxes exist at all, which raises the mean confidence of the survivors, which raises `vision_score`. The interaction is not monotone in the obvious direction; test rather than reason about it.

---

## Physics tuning — `detector_py/pothole_detection.py`

Constructor parameters, so they can be overridden per instance without editing the file:

```python
PotholeDetector(drop_margin=2.0, impact_margin=8.0, freefall_threshold=2.5)
```

| Parameter | Lower it | Raise it |
|---|---|---|
| `drop_margin` (3.0) | Threshold rises toward 9.81 → **more** DROP entries, more noise | Threshold falls → only violent drops register |
| `impact_margin` (10.0) | Milder impacts count → more detections | Only hard impacts count |
| `freefall_threshold` (2.0) | Stricter freefall requirement → fewer detections, higher confidence | Looser → speed breakers may start passing |
| `min_air_time` (0.01 s) | Admits shorter events (4 samples at 400 Hz) | Rejects brief noise |
| `max_air_time` (0.25 s) | Rejects long airborne periods | Admits implausibly deep "potholes" |

The **depth divisor 8.0** is not a constructor parameter — it is inline in `_finalize_event`. Changing it rescales every depth the system has ever produced. Treat as a contract (Rule 4), and read the derivation note in [`10-physics-sensor-pipeline.md`](10-physics-sensor-pipeline.md) first.

**Sampling-rate coupling:** `min_air_time` is meaningful only relative to the sample rate. At 400 Hz, 0.01 s is 4 samples. If real hardware samples at 100 Hz, that is 1 sample and the gate becomes useless — raise it.

---

## Vision tuning

### Confidence
| Range | Behaviour |
|---|---|
| 0.10–0.35 | More detections, more false positives |
| 0.35–0.50 | **Recommended operating range** |
| 0.60–0.90 | Few, high-confidence detections |

Set via the filtered GUI slider, the enhanced GUI slider (fixed 2026-08-19 — it previously did not reach detection), `--conf` on `predict_videos.py`, or the `conf=` argument to `detect_potholes`.

### Road-mask performance
`ROAD_MASK_STRIDE = 10` and `ROAD_MASK_WIDTH = 800` are the two knobs that dominate video throughput.

| Change | Effect |
|---|---|
| Raise stride to 20–30 | Roughly halves segmentation cost; the mask lags reality, bad for fast turns |
| Lower stride to 1 | Per-frame accuracy, several times slower |
| Lower width to 480 | Faster segmentation, coarser mask edges |
| Set `use_road_seg=False` | Fastest; **no false-positive filtering at all** |

Note the filtered GUI does **not** downscale or stride — it segments every frame at full resolution. That is why it is slower and why its masks are cleaner.

### Class keyword sets
`TwoStageDetector.road_include_keywords` / `road_exclude_keywords`, and `_get_drivable_class_ids()` in the filtered GUI. **Check these against your segmentation model's actual `names` dict** — substring matching against the wrong vocabulary is the usual cause of an empty or nonsensical road mask.

Against the shipped 7-class `road_seg.pt` both sets resolve correctly, but `TwoStageDetector` only does so
because its exclude pass runs *after* the include pass: `roadside_object` and `road_obstacle` both match the
include substring `road`, get added, then get subtracted again. The net mask is `visible_road` minus
occluders. **Reordering those two passes would silently admit roadside clutter as road.** Full trace in
[`11-vision-pipeline.md`](11-vision-pipeline.md).

### `_refine_road_mask` geometry (filtered GUI)
Hardcoded component filters: min area `max(300, 0.1% of frame)`, must reach below 78 % of frame height, centroid within 12–88 % horizontally. These encode "the road is a big blob at the bottom centre". A dash-cam mounted unusually high, low, or off-centre will need them adjusted.

---

## Map UI tuning

| Constant | Notes |
|---|---|
| `PROXIMITY_RADIUS_M` (100) | At 60 km/h, 100 m is ~6 s of warning. Raise for testing; raising it far increases repeat-alert churn |
| `PAVE_POLL_INTERVAL_MS` (3000) | Lower for snappier demos; each tick refetches the entire file |
| `MAP_CENTER` | Only used before the first GPS fix |
| Detection-status revert (4000 ms) | `window._rst` timer in `updateStatus()` |
| Toast duration (3500 ms) | `toastTimer` in `showToast()` |
| Live feed cap (10) | The `while (list.children.length > 10)` loop in `addToList()` |
| Theme | Nine CSS custom properties in `:root` — see [`13-map-ui.md`](13-map-ui.md) |

---

## Training hyperparameters

Full preset tables in [`12-vision-training-scripts.md`](12-vision-training-scripts.md). Summary: three presets — `baseline` (100 epochs, lr0 0.01), `aggressive` (150 epochs, lr0 0.02, mixup 0.1), `conservative` (80 epochs, lr0 0.005). All at `imgsz 640`, `batch 16`, AdamW, `seed 42`.

**Do not enable `flipud` or `degrees`.** Road scenes have a fixed gravity orientation; vertical flips and rotations teach the model nothing that occurs in the real input distribution.

RandomForest: `n_estimators=200`, `max_depth=12`. Both in `Model/train_ai_model.py`. Rule 8 applies to retraining either model.

---

## Change checklist

Before changing any value here:

1. **Note the current value** — most have never been re-derived since they were first chosen.
2. **Know the blast radius** — fusion constants affect everything; GUI constants affect one app.
3. **Measure.** For sensor-side changes, run `pytest integration/test_integration.py -v -s` and record
   the **EVENT-level** recall and precision before and after. Baseline as of 2026-08-19: recall 1.00,
   precision 1.00 over 3 events. Ignore the per-sample line — it is printed for continuity only and caps
   at 1/event-width.
4. **Update this file and [`50-changelog.md`](50-changelog.md)** (Rule 5).
5. If the value is part of a contract (depth divisor, feature columns, mask polarity), also update [`20-data-contracts.md`](20-data-contracts.md) (Rule 4).
