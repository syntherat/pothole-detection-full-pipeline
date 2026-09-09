# 40 — Known Issues & Gaps

**Scope:** every bug, mock, dead path, and missing file found in a full scan of the repo.
**Read this before debugging anything.** Most "it's broken" reports in this project are already listed here.
**Last verified against:** commit `3c586b8` (2026-08-19).

Severity: 🔴 blocks a documented workflow · 🟡 wrong behaviour, workaround exists · 🟢 cosmetic or stale docs
Category: **[BUG]** defect · **[MOCK]** deliberate placeholder · **[GAP]** unbuilt · **[STALE]** doc/path drift

---

## 🔴 Blockers

---

## 🟡 Wrong behaviour

### 6. `_toggle_road_seg()` is a no-op — **[BUG]**
**Where:** `app/main_enhanced.py`
Its own comment admits it: *"This doesn't actually change the detector, just a flag."* It only calls `_update_preview()`. The real branch is read at detection time from `use_road_seg_var`, so the checkbox does work — but the handler's name promises something it does not do.

### 7. `test_detection.py` depends on absent directories — **[BUG]**
Looks for images in `input/` and `data/visible_road_seg_public_full/images/val/`, and writes to `OUTPUT_DIR` without creating it.
**Effect:** exits at "No test images found" on a fresh clone.
**Worse than recorded (found 2026-09-06):** that `sys.exit(1)` runs at *import* time, so pytest hits it
during collection and dies with `INTERNALERROR ... SystemExit: 1`, reporting **"no tests ran"**. A bare
`pytest` from the repo root therefore runs zero tests — including the healthy `integration/` suite. Use
`pytest integration/test_integration.py` until this is fixed. It also does not look in
`data/sample_images/`, so the stand-in images added for #2 do not help it.
**Correction (2026-08-19):** the `_full` suffix was previously recorded here as matching no directory any script produces. It is in fact the **real training-dataset directory** for `road_seg.pt` — confirmed from the `data.yaml` path recorded inside the checkpoint. The path is right; the dataset simply is not in this repo. Only the missing `OUTPUT_DIR` creation is a genuine defect.

### 8. `pave_connector.send_to_pave()` is fragile — **[GAP]**
Full read-modify-write per event: O(n²) over a run, not concurrency-safe (single writer only), unbounded file growth, and **no error handling** — a malformed `pave_events.json` raises `JSONDecodeError` mid-run and kills the orchestrator.
**Acceptable for a prototype.** The module docstring names the upgrade path (POST to a Flask endpoint).

### 9. `orchestrator.__main__` runs unbounded — **[BUG]** · partially mitigated 2026-08-19
The file's comment says *"Start small — full 200s @ 400Hz is 80,000 rows"*, then calls `run()` with no limit. 80,000 rows with a YOLO forward pass per triggered row.
**Mitigated:** there is now an argparse CLI with `--limit`, and the no-limit path prints a warning before starting. The default is still unbounded, so **Rule 8 still applies** — pass `--limit`.

### 11. `evaluate_model.py` needs undeclared dependencies — **[BUG]**
Imports `matplotlib` and `seaborn`; neither is in `pothole_detection_app/requirements.txt`. `merge_datasets.py` imports `yaml` (PyYAML), also undeclared.

### 12. Two GPS mocks disagree — **[MOCK]**
- `integration/frame_provider.py` → **Bhopal** (23.2599, 77.4126), a straight northward line at ~1 m/s, unrelated to the CSV's `speed` column.
- `pothole_app_filtered.py::_random_dummy_gps` → **San Francisco** (37.7749, -122.4194), randomised per video.

Events from the two paths would land on opposite sides of the world. Harmless today (they never mix), confusing later.

### 13. `predict_videos.py` defaults outside the repo — **[BUG]**
`--vids-dir` defaults to `<repo_parent>/vids`, i.e. a sibling of the repository. Always pass `--vids-dir` explicitly.

---

## 🟢 Stale paths and docs

### 14. Machine-specific absolute paths — **[STALE]**
| File | Dead path |
|---|---|
| `data/data.yaml` | `C:\Users\palso\OneDrive\Desktop\VITB\epics\pothole_detect_app\data\yolo` |
| `scripts/organize_dataset.py` | `SOURCE_DIR = C:\Users\palso\OneDrive\Desktop\VITB\epics\Pothole Dataset` |

Everything else anchors on `Path(__file__).resolve().parent`. Rule 7: do not add more.

### 15. Subsystem READMEs are partly stale — **[STALE]**
~~Root `README.md` names `EPICS/`, `pothole_detect_app/`, `pothole-map-ui/`, tells you to paste the Maps key into `app.js`, and documents no integration layer.~~ **Rewritten 2026-08-20** — the root README now matches reality: correct folder names, all five subsystems including `integration/` and `carla_sim/`, the `config.js` key flow, an honest status table, and the per-event accuracy figures. It deliberately does **not** reference `context/`, `CLAUDE.md` or `AGENTS.md`, since those are gitignored and absent for anyone who clones.

Still stale: `pothole_detection_app/README.md` duplicates a lot of the old root README, including the model-performance table that presents upstream architecture ranges as if they were measurements of `best.pt` (Rule 6).

~~`pothole_detect_physics/README.md` places `run_detector_on_dataset.py` under `detector_py/`~~ — corrected 2026-08-19 alongside issue #3. The root README's stale folder names remain.

### 16. `index.html` comment names the wrong global — **[STALE]**
The comment says `app.js` reads `CONFIG.GOOGLE_API_KEY`. It reads `window.GOOGLE_API_KEY`, which is what `config.example.js` sets. Code is right, comment is not.

### 17. `sampling_rate_hz` is stored but never used — **[STALE]**
`PotholeDetector.__init__` takes and stores it; no method reads it. The air-time gates are absolute seconds, so the detector is sample-rate-agnostic by accident. See the sampling-rate coupling note in [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md).

---

### ~~28. CARLA mode has no vehicle position, so proximity alerts never fire~~ — RESOLVED 2026-09-06
Resolved by `integration/carla_replay.py`, which publishes contract #12 (`vehicle_position.json`) from a
recorded run's `gnss.csv`, and a 200 ms poller in `app.js` that moves the car marker and calls
`checkProximity()`. The original text follows.

### 28 (original). CARLA mode has no vehicle position, so proximity alerts never fire — **[GAP]** · found 2026-09-06
The dashboard's CARLA mode (added 2026-09-06) draws the town and drops pothole markers, but shows **no car
marker**, and therefore never raises a proximity alert.

Browser geolocation is switched off there on purpose: it would place the vehicle in Bhopal while the CARLA
roads sit near (0, 0), thousands of km apart. Drawing nothing beats drawing something confidently wrong.

The position exists — `drive_and_record.py` already writes `gnss.csv` every tick — it is simply not exposed
to the UI. `pave_events.json` carries only confirmed detections, not a continuous track.

**Effect:** in CARLA mode the dashboard demonstrates detection and mapping, but not the *driver-alert* half of
the product. `checkProximityFromLastKnown()` guards on `if (carMarker)`, so this is a quiet no-op, not a crash.

**Fix options:** have the connector publish a small `vehicle_position.json` alongside the events file and poll
it the same way, or replay `gnss.csv` in the browser against the run's timestamps. Either is a new coupling
point and needs an adapter in `integration/` plus a contract entry (Rules 3 and 4).

## Structural gaps — unbuilt, not broken

### 29. Stage-1 road mask fails on non-dashcam imagery, and the fallback then suppresses detections — **[GAP]** · found 2026-09-06
Measured, not inferred. Running the real cascade over the 17 stand-in photos added on 2026-09-06
(close-up 4032x3024 phone shots of Kerala roads):

| | result |
|---|---|
| `road_seg.pt` produced a usable mask | **0 / 17** images |
| "No road mask generated" fallback fired | **17 / 17** |
| raw `best.pt`, full frame, conf=0.05 — images with any box | **11 / 17** |
| full cascade, conf=0.05 — images with any box | **3 / 17** |
| full cascade, conf=0.35 — potholes confirmed | **1 / 13** |
| clean-road images falsely flagged, any threshold | **0 / 4** |

The documented empty-mask fallback (`two_stage_detection.py:153`, described in
[`11-vision-pipeline.md`](11-vision-pipeline.md) step 7) sets the lower 60 % of the frame to road. In a
dash-camera frame that is a sound guess. In a close-up photo taken standing over a pothole, the subject is
usually *centred or high* in the frame, so the fallback crops it away — which is why the cascade finds
strictly fewer potholes than the raw model does on the same images.

**This is not a defect in the cascade.** It is a domain mismatch between `road_seg.pt` and the stand-in
images that were chosen. Both stages behave as designed. The lesson is about the mock, not the model.

Also checked and **ruled out**: `vision_adapter.VisionSession.confirm()` scores a frame with
`mean(confidences)` rather than `max`, which in principle penalises a frame containing several weak
detections alongside a strong one. Across these 17 images it changes the outcome on exactly one image at
conf=0.50 and nothing at all below that, so it is not what is driving the low recall. Worth revisiting only
if frames start yielding many boxes.

**What this does and does not license you to say.** The pipeline demonstrably runs end to end on real
photographs, loads both checkpoints, and raises zero false positives on clean road. It supports **no recall
claim whatsoever** — 1/13 measures the stand-in images against a road-segmentation model trained for a
different camera geometry, not the detector's ability to find potholes. Rule 6 applies, and #18 still blocks
any real accuracy figure.

**RESOLVED 2026-09-06 by option (a).** The 13 close-ups were replaced with 16 dash-camera frames from
`Ryukijano/Pothole-detection-Yolov8` (roboflow, 640x640, YOLO labels shipped). The four Kerala `none_*`
controls were kept — an all-positive pool would confirm every event, which is the failure the mixed pool
exists to catch.

Re-measured on the new pool:

| | before (close-ups) | after (dashcam) |
|---|---|---|
| frames confirmed at conf=0.35 | 1 / 13 | **10 / 16** |
| controls falsely flagged | 0 / 4 | **0 / 4** |
| labelled potholes localised, IoU >= 0.3 | not measurable | **16 / 33** |

Localisation was scored against the shipped YOLO labels, which is a stricter check than counting hits: where
the detector fires it lands on the real pothole, with per-frame best IoU typically 0.6-0.89. One frame
(`dash_0201`) fires at 0.13 IoU — a detection next to the pothole rather than on it.

**The improvement came from frame geometry, not from road segmentation** — stage 1 still falls back on
20/20 images. See issue #30, which that measurement uncovered.

### 30. `road_seg.pt` never predicts `visible_road`, so stage 1 contributes nothing — **[BUG]** · found 2026-09-06
The two-stage cascade is, in practice, **single-stage plus a fixed lower-60% crop.** Measured, not inferred.

`road_seg.pt` exposes 7 classes: `visible_road`, `vehicle`, `pedestrian`, `shadow`, `vegetation`,
`roadside_object`, `road_obstacle`. Run over the 16 dash-camera frames now in `data/sample_images/`, at a
threshold as low as **conf=0.05**:

| | result |
|---|---|
| frames with any `visible_road` (class 0) prediction | **0 / 16** |
| classes it *does* predict | `roadside_object` (every frame), `vehicle`, `pedestrian`, `road_obstacle`, `shadow` |
| "No road mask generated" fallback fired | **20 / 20** images, both geometries |

So `road_mask.max() == 0` every time and `two_stage_detection.py:153` substitutes the lower 60 % of the
frame. **Every pothole detection this system has ever produced came from that crop, not from road
segmentation.** On dash-camera geometry the crop is a reasonable proxy, which is exactly why this has gone
unnoticed — the cascade produces sane results for the wrong reason.

**This contradicts the claim in `CLAUDE.md`** that two-stage detection is "live... no longer the silent
single-stage fallback it used to be". Corrected there 2026-09-06 (Rule 1: code wins on behaviour).

**Second, separate defect — keyword matching is substring-based.** `_resolve_class_ids()` matches
`road_include_keywords = {"road", "lane", "street", ...}` as substrings against class names, so:

- `roadside_object` contains "road" → **included as road**
- `road_obstacle` contains "road" → **included as road** (then re-excluded, since "obstacle" is in the
  exclude set — the two sets overlap and the exclusion happens to win)
- `roadside_object` contains "road" → **included as road** (then re-excluded — see the correction below)

**Correction, 2026-09-08 (verified by running `_resolve_class_ids` over the checkpoint's 7 names).** The
bullet above previously claimed `roadside_object` is *not* removed by the exclude set, "because that set
lists `"roadside object"` with a **space** while the class name uses an **underscore**", and concluded the
cascade would hunt for potholes *inside roadside objects*. **That is wrong.** `_normalize_class_name()`
does `.replace("_", " ")` before matching, so `roadside_object` → `roadside object`, which matches the
exclude set exactly. Resolved net result over all 7 classes:

| class | include | exclude | net |
|---|---|---|---|
| `visible_road` | yes | no | **ROAD** |
| `roadside_object` | yes | yes | not road |
| `road_obstacle` | yes | yes | not road |
| `vehicle`, `pedestrian`, `shadow`, `vegetation` | no | yes | not road |

**Only `visible_road` is ever admitted, which is correct.** The substring matching is therefore not a
latent hazard, and fix option (a) below is moot. The sole defect is (b): the checkpoint never fires
`visible_road`.

**Fix options:** (a) match class names exactly rather than by substring, and align the underscore/space
mismatch — cheap, correct, and does not fix the real problem; (b) work out why the checkpoint never fires
`visible_road` — wrong `imgsz`, a preprocessing mismatch against its training pipeline, or a genuinely
under-trained class; (c) accept the lower-60% heuristic as the real Stage 1, delete the segmentation stage,
and stop describing the system as two-stage. **Do not do (a) alone and call it fixed** — it would change
behaviour from "empty mask, safe fallback" to "mask covering roadside objects", which is worse.

**Reproduced on CARLA Level B frames, 2026-09-08.** Run over the 45 picked frames of `levelB_vision`
(`carla_sim/out/levelB_vision/vision_results.csv`):

| | result |
|---|---|
| frames with any `visible_road` prediction | **0 / 45** |
| classes it *does* predict | `road_obstacle`, `roadside_object`, `pedestrian`, `vehicle` |
| "No road mask generated" fallback fired | **45 / 45** |

So the behaviour is identical on synthetic frames: empty mask, lower-60% crop, every time. This is now
measured on two independent frame sets (16 real dash-cam, 45 CARLA) rather than one.

**What the checkpoint itself records — read out of `road_seg.pt` on 2026-09-09, no GPU and no dataset
needed.** The `.pt` is a zip; `best/data.pkl` was disassembled with `pickletools` (which parses opcodes
without importing ultralytics), so these are the file's own recorded values, not a re-measurement.

Training arguments:

| | |
|---|---|
| architecture / task | `yolo11s-seg.yaml`, `segment`, `nc=7` |
| `data` | `D:\epics\pothole_detect_app\data\visible_road_seg_public_full\data.yaml` |
| `imgsz` / `batch` / `epochs` | **640** / 10 / 60 (`patience=30`, never triggered) |
| optimiser | AdamW, `lr0=0.003`, cosine off, `close_mosaic=10` |
| resumed from | `...\road_seg_multiclass_small_20260316_074142\weights\last.pt` — this was a **resumed** run |
| augmentation | `mosaic=0.2`, `scale=0.35`, `fliplr=0.5`, `hsv_v=0.3`, `degrees=0`, `perspective=0` |

Validation metrics at the saved (best == final, epoch 60) checkpoint, on the val split of that dataset:

| | box | mask |
|---|---|---|
| precision | 0.5350 | 0.4978 |
| **recall** | **0.3187** | **0.2837** |
| mAP50 | 0.3475 | 0.3016 |
| mAP50-95 | 0.2060 | 0.1463 |

`fitness` 0.3523. The full 60-epoch curves are in the checkpoint too: everything plateaus by roughly
epoch 30 and the last 30 epochs move mAP50(M) by 0.005.

**What that does and does not settle.** It rules out the strongest form of the loading hypothesis —
**this checkpoint is not inert.** It produced masks and scored a non-trivial mAP on its own validation
split, so "the weights are broken" and "the file never had a working segmentation head" are both dead.
It also confirms `imgsz=640`, which is what ultralytics' predict path defaults to, so **there is no
`imgsz` mismatch** between training and how `two_stage_detection.py` calls it.

**It does not settle issue #30**, for one specific reason: **these numbers are 7-class aggregates.**
`visible_road` is one of seven, and a per-class row is what the issue turns on. A model that segments
`vehicle` and `roadside_object` well and `visible_road` not at all would produce exactly this table.
Note also that recall is **0.28–0.32 overall** — this is a weak model even in-distribution, which is
consistent with what it does on dash-cam frames without explaining it.

**The missing number, and how to get it.** `scripts/validate_road_seg.py` (added 2026-09-09) prints the
per-class table, sweeps predict-mode confidence, and measures the in-distribution fallback rate. **It
has not been run** — the `visible_road_seg_public_full` split is gitignored and is on neither this
machine nor, at the recorded path, obviously on the Windows one: the checkpoint names
`D:\epics\pothole_detect_app\...`, an older root than the current `D:\dev\PAVE\...`. Locate the
split first; rebuilding it means re-downloading Cityscapes + ACDC + IDD + Mapillary.

**One hypothesis is already dead.** The confidence gate is not the dash-cam explanation:
`get_road_mask()` passes no `conf` and therefore runs at ultralytics' predict default of 0.25, but the
0/16 and 0/45 measurements above were taken at **0.05** and still found nothing. The gate is worth
measuring in-distribution to see the class's score distribution, not as a candidate fix.

**Do not claim a two-stage architecture in any writeup until this is resolved.** Rule 6.

**How this was handled in the patent disclosure (2026-09-09).** The restriction above was honoured
rather than worked around. §6B of the disclosure describes **two co-equal embodiments** for producing
the road mask — (i) the semantic class algebra, and (ii) the geometric road prior (the lower-60 %
region) — and states that the centroid retention test is defined independently of which produced the
mask. The geometric prior is written as a designed alternative, not as an error path. No sentence
claims that segmentation is operative, and no performance figure is attached to either embodiment.
The claim is drafted so that its independent limitation reads on the mask **however derived**, which
means it reads on the shipped system as it actually behaves. Nothing in the filing asserts something
this issue contradicts.

### 42. `SensorSession.check_row` calls `predict_proba` per row — whole-run analysis takes ~40 min — **[BUG]** · found 2026-09-08
`integration/sensor_adapter.py:52` builds a **one-row `pd.DataFrame` and calls the RandomForest's
`predict_proba` once per sample**. At CARLA's 400 Hz a 60 s run is 24,000 rows, so that is 24,000 separate
sklearn calls, each dominated by per-call Python and validation overhead rather than by the forest itself.

**Measured 2026-09-08:** `analyse_run.py` over a 24,000-row run did not finish in **40 minutes** and was
killed with no output (stdout was block-buffered, so the partial results were lost too). Driving
`PotholeDetector.process_sample` directly instead — same FSM, same order, same inputs — the identical
analysis runs in **1.9 s**. Roughly three orders of magnitude.

**Fixed for `analyse_run.py` only** (2026-09-08): it now imports `PotholeDetector` directly and loops with
`itertuples`, because that script never reads `ai_score`. The FSM is stateful so the loop must stay
sequential, but nothing required a `Series` per row.

**`sensor_adapter.py` itself is unchanged and still has this cost.** `orchestrator.py` goes through
`check_row`, so any full-run cascade pass pays it. The fix there is to batch: `predict_proba` over the whole
frame in one call, then index per row — the classifier is stateless, only the FSM is not. Do that before
running the cascade over any CARLA-length recording.

### 32. `-quality-level=Low` crashes CARLA 0.9.16, and the crash masquerades as a timeout — **[STALE/ENV]** · found 2026-09-06
Environment gotcha rather than a repo defect, recorded because it cost time and would cost it again.

Launching the packaged CARLA 0.9.16 Windows build with `-quality-level=Low` on an RTX 4060 Laptop
(driver 610.62) kills it during startup:

```
LowLevelFatalError [File:Unknown] [Line: 139]
Shader compilation failures are Fatal.
```

**Why it misleads.** The RPC port on 2000 opens *before* the crash. So a client connects successfully, then
blocks on the first real call until it times out, and reports:

```
RuntimeError: time-out of 20000ms while waiting for the simulator,
make sure the simulator is ready and connected to 127.0.0.1:2000
```

That message points squarely at the timeout value. It was in fact a dead process, and `CARLA_TIMEOUT_S` was
raised from 20 to 120 on that false diagnosis before the crash dialog surfaced the truth. The comment in
`config.py` records the correction rather than hiding it.

**Diagnostic rule:** before touching `CARLA_TIMEOUT_S`, check `Get-Process CarlaUE4*`. An empty result means
a crash, not a slow simulator.

**Also:** `./CarlaUE4.exe Town03` is silently ignored on this build — it boots `Town10HD_Opt` regardless.
Load towns with `client.load_world("Town03")` from Python.

**Workaround:** launch at default quality. Verified working:
`./CarlaUE4.exe -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000`

### 33. The AI filter rejects nothing — cascade recall is set entirely by the FSM — **[GAP]** · found 2026-09-06
Measured while adopting stages A and B of `PAVE_v2_change_log_REVIEWED.docx`.

Running the full cascade over the 80,000-row synthetic dataset:

| | raw-feature model | rolling-feature model |
|---|---|---|
| physics detections (rows) | 154 | 154 |
| **AI-confirmed detections (rows)** | **154** | **154** |
| events detected | 153 / 192 | 153 / 192 |
| event recall | 79.69 % | **79.69 %** |

**The AI filter confirms 100 % of what the physics FSM proposes.** It has never rejected a single candidate
on this dataset, so it contributes nothing to the cascade's output.

The consequence is worth stating plainly, because it inverts the obvious reading of the classifier metrics:
improving the classifier's own recall from **80.36 % to 96.23 %** (item 4.3's rolling features, on a grouped
split) changes end-to-end cascade recall by **exactly zero**. The FSM is the sole bottleneck — the 39 missed
events were never proposed to the classifier in the first place.

**Do not report the classifier's recall as a system figure.** They measure different things, and only the
event-level cascade number describes what the product does.

Why it happens: `should_trigger_vision()` gates on the physics decision, and the classifier is applied to
rows the FSM already selected — rows which, by construction, look exactly like the pattern both were built
around. Genuinely testing the filter needs candidates the FSM proposes wrongly, which this dataset barely
contains (spurious firings: 0).

**Fix direction:** this is an argument for the document's own closing recommendation — real recorded data.
A filter that never rejects anything is untested, not perfect.

### 34. Overlapping events silently mislabelled 4.2 % of the positive class — **[BUG]** · FIXED 2026-09-06
`generate_dataset.py` inserts 200 potholes, then 150 speed breakers, both at independent random starts with
no collision check. When a speed breaker landed on a pothole it overwrote `az` — but the pothole's
`label = 1` survived underneath it.

Speed breakers are this dataset's **deliberate hard negative**; the generator's own comment says
*"IMPORTANT: label stays 0 (not a pothole)"*. The overlap taught the model the exact opposite.

Measured on a fresh 80,000-row generation before the fix:

| | |
|---|---|
| rows labelled pothole carrying a speed-breaker signal | **100** |
| share of all positive labels | **4.2 %** |
| mean `az` on those rows | 10.02 (vs 6.41 on clean pothole rows) |
| potholes overwritten entirely | 5 of 200 |

This was invisible before `event_type` existed — there was no way to ask the question. Fixed by clearing
`labels` across a speed breaker's span, so the label follows the signal.

**The committed `synthetic_pothole_dataset.csv` still contains this defect.** It was not regenerated: the
generator was unseeded when that file was written, so regenerating produces different data and invalidates
the pickled model trained on it. Every figure measured against the committed dataset therefore carries a
~4 % mislabelled positive class.

### ~~35. The FSM does not fire on CARLA data~~ — RESOLVED 2026-09-06 ⭐
**The single most important finding of the session.** Two defects, one in the detector and one in the
simulator's pothole model, each of which alone kept the sensor stage silent on CARLA data.

#### Defect 1 — the FSM required a discontinuity that only hand-written data has

FREEFALL used to reset on any sample above `G - 0.5` that had not already cleared the impact gate:

```python
elif az > (G - 0.5):
    self.reset_state()
```

That demands the signal cross from the freefall band to above 19.81 **in one 2.5 ms sample**, skipping the
whole 9.31–19.81 range. `generate_dataset.py` writes exactly that discontinuity by hand — freefall then
impact with nothing between, producing jumps like `0.29 → 22.11`.

| | samples inside the reset band during a pothole |
|---|---|
| synthetic dataset | **4.8 %** |
| CARLA | **33.3 %** |

A continuously-rising physical rebound passes *through* the band and was always discarded. **The detector
could only ever have worked on data shaped like its own generator's output.** This is
`PAVE_v2_change_log_REVIEWED.docx` section 7 — the circularity argument — demonstrated mechanically.

**Fix:** wait instead of resetting, bounded by `max_air_time`. A real CARLA strike recovers through
`… -1.24, 0.00, 0.39, 1.02` then rises `19.33 → 25.07`; the old rule threw it away at **19.33**, one sample
short of qualifying.

**Regression check:** synthetic event recall **79.69 % → 80.73 %**, precision 100 %, spurious firings 0.
The loosened rule slightly *helps* the synthetic set. `pytest integration/test_integration.py`: 9 passed,
1 skipped, unchanged.

**A trap inside the fix.** The first attempt also moved the event's timestamp to the rebound *peak*, which
is arguably more correct physically. It halved cascade recall (80.7 % → 45.8 %) while the physics detector
fired *more* (154 → 156 events). Reason: `pothole_detected` then lands on a row that looks like ordinary
driving, and the RandomForest filter scores it 0 and discards the event. **Measuring only the detector would
have shown an improvement.** The event is emitted on the threshold-crossing sample.

#### Defect 2 — an impulse cannot produce a freefall phase

`PotholeDetector` requires the near-zero phase to last `min_air_time` (10 ms = 4 samples). An **impulse**
gives the wheel a downward kick — a 1–2 sample transient — and no value of `IMPULSE_DELTA_V` changes that.
A real pothole **unloads** the wheel for the entire fall, holding the accelerometer near zero for tens of
milliseconds.

**Fix:** `ImpulseApplier.schedule_unload()` holds a downward force of `mass × g × severity × UNLOAD_FORCE_SCALE`
at the striking wheel for `UNLOAD_TICKS`, cancelling the suspension's upward reaction, then releases so the
spring re-compresses. Drop, freefall and impact all come from vehicle dynamics.

Force and duration are **coupled** — see the sweep table in `21-configuration-and-tuning.md`. Working pair:
`UNLOAD_FORCE_SCALE = 3.2`, `UNLOAD_TICKS = 19`.

**Result:** the sensor stage now fires on CARLA recordings. A representative strike:

```
9.77 | -7.99 -13.15 ... -2.14 -1.53 -1.24 0.00 0.39 1.02 | 19.33 | 25.07 24.36 23.84
     |  DROP           |  FREEFALL, 6 samples in |az|<2  |  wait |  IMPACT
```

#### What this does and does not establish

It establishes that the sensor stage detects potholes in an environment it did not generate, which was
untrue this morning and is the first genuinely independent evidence the physics stage works.

It does **not** establish an accuracy figure. Level A still has no hole in the road mesh, so **vision remains
untestable** — the camera records clean tarmac. #18 still stands. Markers from a Level A replay are labelled
`sensor (CARLA replay)` for exactly this reason.

**Note for anyone considering Level B:** carving real geometry would NOT have fixed Defect 1. A real hole
produces a continuously-rising rebound, which the old FSM discarded just as surely as it discarded CARLA's.
The detector fix was needed regardless, and finding it cost hours rather than the day a source build would
have taken.

### 43. No cross-vehicle aggregation — a second vehicle's non-detection is invisible — **[GAP]** · found 2026-09-09
Asked during the patent work: *if vehicle A confirms a pothole and vehicle B does not, what happens?*
**Nothing happens.** Traced through the code, not inferred:

| | |
|---|---|
| `PotholeCandidateEvent` (`schema.py`) | has **no vehicle identifier field at all** |
| `orchestrator.py:60` | `if not event.sensor_triggered: continue` — a non-triggering row leaves no record |
| `orchestrator.py:95` | `if event.final_decision: send_to_pave(event)` — a rejected event writes nothing |
| `pave_connector.py:31` | `events.append(record)` — appends unconditionally, never merges or dedupes |
| `app.js:517` | dedupes on `event_id`, which is unique per event — two cars over one hole draw **two markers** |

So vehicle A's positive becomes a permanent marker, vehicle B's negative is never recorded anywhere,
and nothing is ever removed or decayed.

**Why this is not simply a bug to fix.** The two observations are not symmetric. A positive means
something physically happened to the suspension *and* the camera agreed. A negative has at least five
causes, and only one of them means the pothole is gone:

1. B never drove over it — wheel track is ~1.5 m inside a ~3.5 m lane
2. B was slower — the FSM's gates are speed-coupled, and below the critical speed the wheel may not
   lose contact at all
3. B has different suspension — `Aeff = g(1 + ms/mu)` is explicitly vehicle-specific
4. B's camera saw less — night, rain, occlusion, standing water
5. **The pothole was repaired** ← only this one justifies removing the marker

Absence of a strike is not evidence of absence of a pothole. Treating the two as symmetric votes would
degrade the map every time a car changed lanes.

**Design direction, in increasing cost:**
- **Positive-only accumulation.** Ignore negatives. Cluster confirmed events spatially (~10-15 m, since
  consumer GPS error of 3-5 m already exceeds a lane width). Raise confidence on repeat confirmations,
  decay over time. Repair is then handled implicitly. No negative logic needed.
- **Traversal-gated negatives.** Count a negative only where the vehicle demonstrably passed through
  the anomaly footprint at a speed where detection was expected. Needs vehicle identity plus
  lane-level positioning.
- **Per-vehicle normalisation.** Use the quarter-car parameters to derive the `az` a given vehicle
  should produce for a given depth, making cross-vehicle sensor evidence commensurable. Leans on the
  project's strongest existing asset.

**Missing to do any of it:** vehicle ID in the schema · spatial clustering · confidence accumulation ·
decay/expiry · negative records · real GPS (currently mocked, `README.md:130`).

**Patent note — the design above is already claimed, by VIT.** `IN 202241069806` (VIT University) is
**GRANTED**, and its claim 1 was read in full on 2026-09-09. It recites an IMU-only system whose cloud
platform *"marks the position of the potholes using all the data points with respect to the particular
pothole"*, reports **"the confidence of that pothole being there"** along with intensity, warns the
user of incoming potholes in advance, caches locally when there is no reception, and — the part that
matters most here — uses **"the clustering technique … to deal with GPS and data inaccuracies in crowd
sourced data."**

That is the positive-only spatial-accumulation design recommended above, claimed almost point for
point: spatial clustering to absorb GPS error, accumulated confidence per anomaly, advance warning.
**Read that claim set before writing any multi-vehicle aggregation code.** It is held by a sister
campus of this project's own applicant, which makes it an institutional question as well as a legal
one.

Also occupied: `US10967862B2` (Uber/Aurora) claims a **network computing system** that receives sensor
logs from first vehicles and transmits **targeted labels on an updated localization map** to second
vehicles — fleet-level anomaly sharing.

**Per-vehicle suspension normalisation** — deriving the expected `az` for a given vehicle from its
quarter-car parameters so cross-vehicle sensor evidence becomes commensurable — remains the least
occupied direction, and leans on this project's strongest asset. It must be built before it can be
claimed (Rule 6).

### 18. No frame↔sensor synchronisation — **[GAP]** ⭐ the big one
The pipeline's foundational assumption — that a camera frame can be matched to a sensor row by time — has no implementation. `frame_provider.get_mock_frame()` picks an arbitrary stand-in image.

**Everything downstream is unvalidated because of this.** `vision_score`, `final_confidence`, and every fusion weight are measured against images that have no relationship to the sensor event. Until real time-synchronised vehicle data exists, **no vision-stage or fusion accuracy claim is meaningful** (Rule 6).

This is the single highest-value thing to build. **The mechanism to close it is now written** — see
[`15-carla-testbed-plan.md`](15-carla-testbed-plan.md). `carla_sim/` records a `frame_index.csv` on CARLA's
shared timebase, and `integration/carla_frame_provider.py` looks frames up by timestamp instead of by
`event_id`. The lookup logic is tested; **nothing has been recorded yet**, so the gap is not closed in
practice until a real run exists.

### 20. No real sensor data — **[GAP]**
Everything is synthetic: 400 Hz, `speed` resampled per row as white noise rather than a speed profile, injected events placed with **no collision check** (so they can overlap and corrupt each other's signatures), and a generated drop range `U(4,7)` that straddles the FSM's `6.81` threshold. See the interaction note in [`10-physics-sensor-pipeline.md`](10-physics-sensor-pipeline.md).

### 21. No live camera or live sensor input — **[GAP]**
Everything is file-based: a CSV, image files, video files. There is no streaming path, no device capture, no real-time loop.

### 22. Fusion weights are unfitted — **[GAP]**
`0.4 / 0.6` and the `0.5` cut are hand-chosen. Not validated against labelled data — and cannot be until #18 is resolved.

### 23. Two GUIs, diverging — **[GAP]**
`main_enhanced.py` (batch, export, monitoring) and `pothole_app_filtered.py` (realtime video, class filters, event metadata) share **no code** and have separate colour constants, separate model loading, and separate road-mask implementations that behave differently. Both are actively referenced — `run_app.py` launches one, `quick_start.bat` and the READMEs launch the other. Nobody has decided which is the product.

### 24. No CI, no linting, no automated checks — **[GAP]**
`integration/test_integration.py` is the only test file in the repo. Nothing runs it automatically.

---

## Accuracy claims — what you may and may not say

Per Rule 6:

| Claim | Status |
|---|---|
| README model performance tables (mAP 55–80 % etc.) | **Indicative upstream ranges for YOLO sizes.** Not measurements of `best.pt`. Never quote as measured |
| `train_ai_model.py` accuracy print | Real, but on a **3 %-positive synthetic** dataset where all-zeros scores 97 %. Read precision/recall, not accuracy |
| `run_detector_on_dataset.py` "Accuracy" | Mislabelled — it computes `TP/(TP+FP+FN)`, the **Jaccard index**, over classifier predictions, unrelated to the FSM detections printed above it |
| `test_stage1_accuracy_against_real_labels` — **precision** | ✅ Meaningful. Measured 1.00 (TP=3, FP=0) on 2000 rows |
| `test_stage1_accuracy_against_real_labels` — **event recall** | ✅ **3/3 = 1.00** on the first 2000 rows. Computed by the test since the #25 fix |
| The same test's **per-sample** recall line | ⚠ Printed for continuity only. Caps at 1/event-width (0.083 here). **Never quote it as a quality signal** — the output labels it as such |
| Any vision-stage or end-to-end accuracy | **Does not exist.** Blocked on #18 |

---

## Suggested priority

| Order | Item | Why |
|---|---|---|
| 1 | #18 frame↔sensor sync | Highest value overall, largest effort — needs real vehicle data |

---

## Resolved

Kept here with their original numbers so that references elsewhere (and in the changelog) stay valid.
**Numbers are never reused or renumbered.**

### 41. FBX export scale for CARLA must be 1.0, not 100 — **[BUG]** · FIXED 2026-09-08
The Level B plan says "Export FBX; mind metres-vs-centimetres (set FBX scale 100)". That is correct for a
**manual** UE import. It is **wrong for CARLA's import path**, and following it produced tiles 100x oversized.

`Util/BuildTools/Import.py` hardcodes `"bConvertSceneUnit": 1` in the generated import settings, so UE
performs a metres->centimetres conversion of its own. Exporting from Blender at `global_scale=100` makes the
conversion happen **twice**: a 1.6 m tile arrived in-engine at **160 m x 160 m x 14 m**, measured by spawning
it and reading `actor.bounding_box`.

**Fix:** author in metres and export with `global_scale=1.0`
(`carla_sim/assets/generate_pothole_meshes.py`). Verified: tiles then measure 1.6 m in-engine and their
bowls measure their authored depths.

**The wider lesson — checking the artifact was not enough.** `verify_pothole_meshes.py` re-imported each FBX
and reported "160.0 cm across, the metres-vs-centimetres trap avoided". That check *passed* while the assets
were 100x wrong, because **an FBX is internally consistent at either scale**. Only the size AFTER UE's
conversion mattered, and only an in-engine spawn could measure it. Verify artifacts *in their real context*,
not in isolation.

### 40. `make import` runs Import.py via a broken invocation and reports success — **[TRAP]** · found 2026-09-08
`Util/BuildTools/Windows.mk` invokes the importer as a bare command with a **forward-slash** path, relying on
the Windows `.py` file association:

```make
import: server
	@"${CARLA_BUILD_TOOLS_FOLDER}/Import.py" $(ARGS)
```

cmd does not resolve that. The result: `make import` builds the LibCarla server, **produces not one line of
Import.py output, imports nothing, and exits 0.** The `Import/` folder is left untouched and no asset appears
in `Content/`.

**Working invocation** — call the interpreter explicitly, from the CARLA root, with `UE4_ROOT` set and the
usual environment (see the build recipe in [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md)):

```bat
cd /d D:\dev\carla-source
python Util\BuildTools\Import.py
```

**Two further traps in the same path, both of which also return success at the point of failure:**

1. **`KeyError: 'size'`.** A prop entry in the package JSON needs a `size` from CARLA's `EPropSize` enum
   (`Tiny`/`Small`/`Medium`/`Big`/`Huge`, defined in `Actor/PropParameters.h`). Omit it and the FBX **still
   import successfully**, then `generate_package_file()` crashes — so the meshes exist but no
   `<Package>.Package.json` is written and the props never become spawnable `static.prop.*` blueprints.
2. **Registered asset paths can point at assets that do not exist.** `Import.py` derives the object path from
   the *FBX file name*, assuming a single-mesh FBX:
   `/Game/<pkg>/Static/<tag>/<name>/<fbx_basename>.<fbx_basename>`.
   An FBX with multiple root nodes — which any FBX carrying `UCX_` collision bodies has — imports as
   `<fbx_basename>_<mesh_name>.uasset`, so the registered path is wrong and the prop silently fails to spawn.
   Fix with `carla_sim/assets/fix_package_paths.py`, which rewrites the paths from what is actually on disk.
   **Re-run it after every import**, because Import.py regenerates the file each time.

**Verify an import by artifact, never by exit code:**
```bat
dir /s /b Unreal\CarlaUE4\Content\<Package>
```
Expect one `.uasset` per prop plus `Config\<Package>.Package.json`. One `.uasset` per prop (rather than one
per mesh in the FBX) is also the sign that UE consumed the `UCX_` bodies as collision instead of importing
them as separate meshes.

### 38. `Build.bat UE4Editor` does NOT build the engine's Program targets — **[TRAP]** · found 2026-09-08
Building the engine with

```
Engine\Build\BatchFiles\Build.bat UE4Editor Win64 Development
```

produces `UE4Editor.exe` and its 417 DLLs, reports **4238/4238 actions, zero errors**, and looks complete.
**It is not.** The engine's *Program* targets are separate and are NOT built by that command:

| Program | Consequence if missing |
|---|---|
| **`ShaderCompileWorker.exe`** | **The editor cannot compile a single shader.** It starts, throws `Unable to launch .../ShaderCompileWorker.exe - make sure you built ShaderCompileWorker.`, and exits |
| `UnrealLightmass.exe` | cannot build lighting |
| `UnrealPak.exe` | cannot package — needed for `make package` at Level B P1 |
| `CrashReportClient.exe` | no crash reporting (cosmetic) |

Building `UE4.sln` — the route CARLA's own docs describe — builds these as a side effect. The narrower
`Build.bat UE4Editor` route does not, and nothing warns you.

**Fix (a few minutes each, ~157 actions for ShaderCompileWorker):**
```
Engine\Build\BatchFiles\Build.bat ShaderCompileWorker Win64 Development -WaitMutex
Engine\Build\BatchFiles\Build.bat UnrealLightmass     Win64 Development -WaitMutex
Engine\Build\BatchFiles\Build.bat UnrealPak           Win64 Development -WaitMutex
```

**Verify the engine, not just the editor:**
```
dir Engine\Binaries\Win64\ShaderCompileWorker.exe UnrealLightmass.exe UnrealPak.exe UE4Editor.exe
```
All four must exist. Checking only for `UE4Editor.exe` is what let this through on 2026-09-07.

**Related diagnostic trap:** `UE4Editor.exe` being in the process list does **not** mean the editor works. It
sat at 1.2 GB working set while already having failed. And its splash bar parks at
`Initializing. 39%` for the entire shader-compilation phase — a frozen percentage there is normal, not a
hang. Judge by `ShaderCompileWorker` process count and their accumulating CPU: 13 workers consuming ~126 s
of CPU per 12 s of wall clock is a healthy first-run compile.

### 39. Stale `CMakeCache.txt` blocks a generator change across the whole CARLA Build tree — **[TRAP]** · found 2026-09-07
After switching CARLA's build back to the `Visual Studio 17 2022` generator, `make launch` died with

```
CMake Error: Error: generator : Visual Studio 17 2022
Does not match the generator used previously: NMake Makefiles
Either remove the CMakeCache.txt file and CMakeFiles directory or choose a different binary directory.
```

CMake refuses to change generator inside an existing binary directory. Fixing the two directories that had
already failed was not enough — **every** build dir configured under the old generator is a latent blocker.

**Sweep them all at once:**
```bash
find Build -name CMakeCache.txt | while read c; do
  grep -m1 '^CMAKE_GENERATOR:INTERNAL=' "$c"; dirname "$c"
done
```

On 2026-09-07 that found six NMake caches: `libcarla-visualstudio` (the one that broke `make launch`), plus
dormant ones in `gtest-src/build`, `proj-src/build`, `recast-src/build`, `rpclib-src/build`.

**`zlib-source/build` is the exception — leave it.** `install_zlib.bat` hardcodes `-G "NMake Makefiles"`, so
an NMake cache there is correct.

Deleting a dependency's *build* dir is safe: the installer scripts skip work based on the *install* dir, so
nothing is rebuilt unnecessarily.

### 37. CARLA's Windows build scripts report SUCCESS when they have failed — **[TRAP]** · found 2026-09-07
**Do not trust `make`'s exit code on Windows. Verify the artifact.**

Two of CARLA 0.9.16's batch scripts print a success banner and return 0 even when the step they wrapped
failed outright:

- **`BuildOSM2ODR.bat`** printed
  `OSM2ODR has been successfully installed in "...\PythonAPI\carla\dependencies\"`
  immediately after its CMake configure had died with
  `Generator "NMake Makefiles" does not support platform specification, but platform "x64" was specified`
  followed by `CMAKE_C_COMPILER not set`. No header was installed.
- **`BuildPythonAPI.bat`** printed
  `Carla lib for python has been successfully installed in "...\PythonAPI\carla\dist"`
  after the wheel build had failed with `ERROR Backend subprocess exited when trying to invoke build_wheel`.
  `dist/` was **empty**.

In both cases `make PythonAPI` exited **0**. Twice in a row a run "succeeded" and produced nothing.

**How to actually verify `make PythonAPI`:**
```
dir  D:\dev\carla-source\PythonAPI\carla\dist          rem must contain carla-0.9.16-cp312-...whl
if exist D:\dev\carla-source\PythonAPI\carla\dependencies\include\OSM2ODR.h  echo OSM2ODR ok
```
A good build produces `carla-0.9.16-cp312-cp312-win_amd64.whl` (~5.4 MB) containing
`carla/libcarla.cp312-win_amd64.pyd` (~17.7 MB). An empty `dist/` means failure regardless of exit code.

Related: the same "don't trust the wrapper" rule applies to `Update.bat`, whose failure path deletes the
whole content folder — see the changelog entry for 2026-09-07.

### ~~36. Level B toolchain prerequisites~~ — RESOLVED 2026-09-07
Audited 2026-09-06 before committing to a ~165 GB source build, and fully cleared within a day. Recorded
because it gates Level B ([`15-carla-testbed-plan.md`](15-carla-testbed-plan.md)), the top-priority item,
and because two of the diagnostics below are traps that would cost time again.

**Final state — every toolchain prerequisite met:**

| Requirement | Measured |
|---|---|
| Epic / CARLA GitHub access | `git ls-remote` on the **private** `CarlaUnreal/UnrealEngine` returns HEAD `2ac0528`; branches `4.26` and `carla` present; `gh` authed as `syntherat` with `repo` scope. This was the plan's highest-likelihood risk |
| CARLA source tag `0.9.16` | exists on `carla-simulator/carla`, matches the installed client and packaged simulator |
| Disk | D: 306.5 GB free vs ~165 GB needed. C: 25.7 GB — build on D: |
| C++ workload | `Microsoft.VisualStudio.Workload.NativeDesktop` 17.14.37314.3 |
| **MSVC toolset** | v142 (`14.29.30133`) was installed here — **but see the correction below: the engine actually built with v143 (`14.44.35207`) from a VS 2022 *BuildTools* instance this audit never scanned** |
| **`make` 3.81** | `C:\GnuWin32\bin\make.exe`, **GNU Make 3.81** (2006 build). Machine PATH **position 0** |
| CMake / Git / Python x64 | 3.31.11 / 2.49.0 / 3.12.10 AMD64 |
| Windows 8.1 SDK, .NET 4.6.2 pack, 7-Zip | present; 7-Zip 26.03 |

**Two traps worth keeping:**

1. **`vswhere -requires <componentId>` misreported during this audit — and 2026-09-07 found the reason.**
   It was almost certainly `NoDefaultCurrentDirectoryInExePath=1` (see issue #37 and the build recipe in
   [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md)). Once that variable was cleared, vcvarsall stopped
   printing `'vswhere.exe' is not recognized`, began reporting its true version (`v17.14.37`, not the generic
   `v17.0`), and CMake's Visual Studio generator found the compiler immediately. Original note follows:
   It called
   `...VC.14.29.16.11.x86.x64`, `...VC.v142.x86.x64` and `...Windows10SDK.19041` ABSENT while a working
   v142 `cl.exe` sat on disk. `-property` queries (installationPath, displayName) are fine. **Diagnose from
   `VC\Tools\MSVC\<ver>\bin\HostX64\x64\cl.exe` and its version banner** — a 19.29.x banner means v142.
2. **The chocolatey `make` 3.81 package is broken and cannot be fixed by retrying.** It downloads GnuWin32's
   `make-3.81.exe`, saves it as `...Install.zip`, then hands that to 7-Zip as a zip. It is an **Inno Setup**
   executable (`MZP` header), so extraction always fails — and 7-Zip cannot unpack Inno regardless.
   The download itself is valid. **Working route:** run the installer directly with Inno silent flags,
   `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="C:\GnuWin32"`. A copy is kept at
   `D:\dev\tools\make-3.81-setup.exe`.
   Install to a **space-free path** — `make` and `C:\Program Files (x86)` interact badly.
   The GnuWin32 dir must go on the **Machine** PATH, not User: Windows composes System entries before User
   ones, so a User entry cannot outrank a system-level shim. Machine PATH backup: `D:\dev\tools\path-backup.txt`.

**Blender — CORRECTED 2026-09-08: it IS installed.** `D:\Program Files\Blender Foundation\Blender 5.2\`
(5.2.0 LTS) and `Blender 4.4\` (4.4.1), neither on PATH. The 2026-09-06 audit reported it absent because it
only checked `C:\Program Files\Blender Foundation` and PATH. **This is the same search error that missed the
VS 2022 BuildTools install.** Query the uninstall registry keys
(`HKLM/HKCU\...\CurrentVersion\Uninstall\*`, plus `WOW6432Node`) before declaring any Windows program
absent — that finds non-default install locations immediately. It is needed to author the Level B
pothole meshes, not to build CARLA, so it does not block the clone or the compile. Tracked in the Level B
section of [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md).

**All three formerly-unverified items — SETTLED 2026-09-07 by a successful engine build.**
`Build.bat UE4Editor Win64 Development` completed **4238/4238 actions in ~89 min with zero errors**:
1. ~~**v142 inside VS 2022 satisfies UE 4.26's UnrealBuildTool.**~~ **CORRECTED 2026-09-07 — this was wrong.**
   UBT never used v142. Its own build log says:
   `Using Visual Studio 2019 14.44.35228 toolchain (C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207)`
   — that is **v143**, from a **BuildTools** instance the audit never scanned. It checked Professional under
   `%ProgramFiles%` and looked for a `2019` folder, but never `%ProgramFiles(x86)%\Microsoft Visual Studio\2022`.
   CARLA agrees: `Windows.mk` passes `--boost-toolset msvc-14.3`. **Build LibCarla with v143** or it is
   ABI-incompatible with the plugin. The v142 component installed on 2026-09-06 was probably never needed.
2. **Windows 10 SDK 10.0.26100 is acceptable** — the worry that a 4.26-era build wanted ~10.0.18362 did not
   materialise.
3. **Absent `clang` is irrelevant on Windows.** The build never needed it.

**Discovered during that build — worth keeping:** Windows Defender throttled the Unreal tooling by roughly
four orders of magnitude. `Setup.bat`'s dependency check ran at **38 KB/s** with 3.4 % CPU, no TCP
connections and no disk reads; after an exclusion was added for the engine tree the same process hit
**415 MB/s** at 88 % CPU. Symptom: low CPU, high "IO Other Operations/sec", almost no data moving.
Add an exclusion for the CARLA source tree too. Separately, `Setup.bat`'s download can hang with the process
still alive (froze at 90 %, 0.00 MiB/s, 604 identical progress lines) — kill `GitDependencies` and re-run,
which resumes rather than restarting.

**Settled 2026-09-07 — the prop shortcut does not exist.** All 99 `static.prop.*` blueprints were probed
with a `world.cast_ray` grid: **none has a usable cavity**, collision volumes behave as sealed convex hulls,
and no prop can make a true hole anyway because the road surface stays put. Full measurement in the Level B
section of [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md). The same run verified P0 item 3 (99/99
props spawn). The source build is unavoidable.

### ~~31. `orchestrator.py` could not start — three defects in the entry point~~ — RESOLVED 2026-09-06
Found by running it. `python integration/orchestrator.py --limit 40000` died immediately with
`NameError: name 'argparse' is not defined`. Behind that were two more, each of which would have fired next:

| # | Defect | Effect |
|---|---|---|
| 1 | `argparse` used in `main()`, never imported | `NameError` on the first line of `main()` — the CLI could never run |
| 2 | `main()` called `run(limit=…, dataset_path=…, frame_provider=…)`; `run()` accepted only `limit` | `TypeError` — `--dataset` and `--carla-run` were unreachable |
| 3 | `skipped_no_frame` printed in the summary, never assigned | `NameError` at the end of every otherwise-successful run |

Together these mean **the orchestrator CLI had never been executed successfully.** Defect 2 is the
`--carla-run` path, which is the command `04-current-state.md` tells the next person to run after recording
a CARLA run — so it sat directly across the project's planned next step.

The shape of it suggests a half-applied change: the provider abstraction and the skip counter were written
into `main()`, into the summary text, **and into `14-integration-layer.md`**, but never into `run()` itself.
Notably the context file documented the correct signature all along — Rule 1 working as intended, with the
docs recording design intent and the code being the thing that had drifted.

**Fix:** imported `argparse`; gave `run()` the `dataset_path` and `frame_provider` parameters the docs
already specified; routed frame and GPS lookups through the provider interface that `MockFrameProvider` and
`CarlaFrameProvider` already shared; initialised the counter. Added a separate `skipped_no_gps`, because
`CarlaFrameProvider.get_gps()` returns `None` outside its recorded window and a positionless event would put
a marker at (0, 0) — the same ocean problem CARLA map mode exists to avoid.

Nothing outside `integration/orchestrator.py` changed; no module imports from it, and `test_integration.py`
imports `get_mock_frame`/`simulate_gps` from `frame_provider` directly, so swapping the orchestrator's own
import to `MockFrameProvider` touched nothing else.

### ~~27. joblib unpickling uses a numpy API deprecated in 2.5~~ — RESOLVED 2026-09-06
Fixed upstream, not by us. The original report was **1602** `DeprecationWarning`s from
`joblib/numpy_pickle.py:207` under `joblib 1.5.3` + `numpy 2.5.1`.

Re-measured 2026-09-06 on `joblib 1.6.0` + `numpy 2.5.2`: `pothole_ai_model.pkl` loads as a
`RandomForestClassifier` with **zero** warnings of any category — checked by counting
`warnings.catch_warnings(record=True)` entries under `simplefilter('always')`, not by eyeballing stderr.

**No numpy pin is needed.** The previously proposed `numpy<2.5` would now be an unnecessary constraint — do
not add it. The floor that actually matters is `joblib >= 1.6.0`; `pothole_detect_physics/requirements.txt`
lists `joblib` unpinned, which resolves correctly today but does not *guarantee* it.

### ~~2. `data/sample_images/` does not exist~~ — RESOLVED 2026-08-20
Three changes, because the folder alone was never the whole problem:

- **The crash became a skip.** `MockFrameProvider.get_frame()` now returns `None` when no images exist,
  matching `CarlaFrameProvider`'s interface, so the orchestrator's skip path handles it and the run finishes
  with a count instead of dying on its first triggered event. `get_mock_frame()` keeps its raising contract.
- **Discovery fallbacks.** `frame_provider.CANDIDATE_IMAGE_DIRS` searches `sample_images/`, then `input/`,
  then the `dataset_v3` and `dataset_v2` test/val splits, using the first that contains images. Anyone who has
  built a training dataset now gets a working demo with no configuration.
- **The folder ships with instructions.** `pothole_detection_app/data/sample_images/README.md` explains what
  to put there and — importantly — what a mock run does and does not prove. Root `.gitignore` narrowed from
  the whole directory to its contents, so the README is tracked while photos stay ignored.

The images themselves are still the user's to supply; they cannot be invented (Rule 6). What is fixed is that
their absence is now a clearly-reported degradation rather than a crash.

Verified: with no images, `get_frame` returns `None` twice and warns once, while `get_mock_frame` still
raises; with images in a fallback dir, discovery finds exactly the 3 image files (excluding a `.txt`) in
sorted order; `sample_images/` takes precedence over the fallback; `git check-ignore` confirms README tracked
and `.jpg` ignored. Two regression tests added; full suite 8 passed, 2 skipped.

### ~~10. `hash()` is salted per process~~ — RESOLVED 2026-08-20
Fixed alongside #2, in the same function. `get_mock_frame()` now indexes with
`int(hashlib.md5(event_id).hexdigest(), 16) % len(pool)` instead of `hash()`, so its "reproducible" docstring
is finally true across processes and not merely within one. The pool is also sorted now — `Path.glob()` order
is filesystem-dependent, which was a second, quieter source of the same non-determinism.

Verified by running `get_mock_frame('fixed-event-id')` in three subprocesses under `PYTHONHASHSEED` 0, 1 and
12345: identical result every time. A regression test covers it (skips when no images are configured).

### ~~19. The filtered GUI's events never reach the map~~ — RESOLVED 2026-08-20
Added `integration/filtered_gui_adapter.py`, which translates `events.jsonl` records into contract #7 and
publishes them. One-shot or `--watch` to follow the file live while the GUI writes it.

Two things had to change first, because the adapter could not bridge the gap honestly on its own:
- **`events.jsonl` had no confidence.** `pave_events.json` requires one, and `potholes_detected` is a count,
  not a confidence — deriving one from it would have been inventing a number (Rule 6). The detector already
  computed per-box confidences in `_detect_frame` and threw them away, so it now keeps them and logs their
  mean. That is a defect fix in the GUI's own log, not a change made to serve integration (Rule 3).
- **`app.js` ignored `detected_by`** and hardcoded `'Both'`, so camera-only detections would have claimed to
  be sensor+vision. It now reads the field, and tolerates a `null` confidence — which previously threw inside
  the loop and aborted every remaining event in the batch.

`pave_connector` gained `send_record_to_pave(record)` so both producers share one writer and contract #7 stays
defined in a single place.

Verified end to end against a fabricated log: field mapping, exact contract #7 key set, legacy records with no
confidence passing through as `null`, records with no coordinates dropped rather than guessed, a truncated
final line ignored (which happens during live writes), idempotent re-runs, new events picked up on a later
pass, both `run_dir` and direct `events.jsonl` paths accepted, and cascade records still writing correctly
alongside GUI ones. The `app.js` mapping was checked in node across five event shapes; none throws.

**Not verified:** the GUI was not run, so the confidence values now being logged are unexercised — `cv2`,
`ultralytics` and `torch` are not installed here.

### ~~12b. `download_road_model.py` would now overwrite the real model~~ — RESOLVED 2026-08-19
The script now **refuses to overwrite an existing output file** and exits 1 with an explanation of what is
already there and why replacing it would be a downgrade. Two escape hatches: `--output <path>` to keep both,
`--force` to overwrite deliberately (which also writes a `.bak` first).

Three supporting changes:
- **The existence check runs before the download**, not after, so a refused run costs no network fetch and
  cannot half-finish. The `ultralytics` import moved inside the function below that check, which is why
  `--help` and the abort path work even without ultralytics installed.
- **Corrected the docstring and output.** It claimed the weights were "trained on Cityscapes/COCO and can
  segment roads". They are COCO-only, and **COCO has no road class** — that claim is what made the script
  look safe to run. It now says plainly that this is a downgrade and points at
  `prepare_visible_road_public_dataset.py` + `train_multiclass_road_seg.py` for a real model.
- Replaced the non-ASCII status glyphs, which raise `UnicodeEncodeError` on a cp1252 Windows console when
  stdout is redirected.

Verified by execution: with `road_seg.pt` present the default run aborts with exit 1 and the model's SHA-256
is byte-identical before and after; `--help` works with no ultralytics installed; `--output <fresh path>` and
`--force` both get past the guard (reaching the ultralytics import, which is absent here — proving the check
runs first); no file was created or clobbered in any case.

### ~~3. `Model/run_detector_on_dataset.py` has a broken import~~ — RESOLVED 2026-08-19
Added a `sys.path.insert` for `detector_py/` before the import, matching the pattern
`integration/sensor_adapter.py` already uses. The file stays in `Model/`; `pothole_detect_physics/README.md`
was corrected instead — it listed the script under `detector_py/` and told you to run it from there.

Considered and rejected: moving the file into `detector_py/`. That would also have fixed it and matched the
old README, but it is a bigger change than the bug warrants (Rule 11). Still a reasonable future tidy-up —
the script arguably belongs beside the detector it runs.

Verified: the import resolves and `PotholeDetector()` instantiates through the new path; the bare import
still fails from `Model/`, confirming the fix is load-bearing.

### ~~4. `quick_start.bat` chains mismatched dataset versions~~ — RESOLVED 2026-08-19
Inserted `merge_datasets.py` as step 2, so the chain is now
organize (to `dataset_v2`) then merge (to `dataset_v3`) then train (reads `dataset_v3`). Renumbered to 5 steps.

Two related fixes in the same file:
- `evaluate_model.py` is now called with `--data data\dataset_v3\data.yaml`. It defaults to `dataset_v2`, so the
  pipeline was training on v3 and evaluating on v2 — the same mismatch one step later.
- **`organize_dataset.py` returns exit code 0 when its source directory is missing or empty**, so
  `if errorlevel 1` never caught it and the batch marched on to a confusing "Data config not found" during
  training. Both dataset steps are now followed by an explicit `if not exist ...data.yaml` check that fails
  where the problem actually is. A header comment points at the hardcoded `SOURCE_DIR` (issue #14), which is
  the usual root cause.

Verified structurally: `if (` blocks balanced, no bare parens inside block echoes, no unescaped redirects.
**Not verified by execution** — running it would start a 1-3 hour training run (Rule 8).

### ~~5. The `_conf` import trap~~ — RESOLVED 2026-08-19
`utils.py` gained `get_conf_threshold()`, the read-side counterpart to `set_conf_threshold()`.
`main_enhanced.py` and `enhanced_utils.py` no longer import `_conf`; all six call sites now call the
accessor, so they read the live value. Both import blocks carry a comment explaining why importing the
float is wrong, to stop anyone "simplifying" it back.

`enhanced_utils.py` also stopped importing `_model` — same by-value trap (it was pinned at `None`), and it
was never used; `load_model()` was already being called correctly.

Verified by importing the real `utils.py` with stubbed heavy deps and driving `set_conf_threshold()` through
0.35 / 0.10 / 0.60 / 0.90: the old `from utils import _conf` pattern stayed pinned at 0.35 throughout, the
accessor tracked every change. An AST check confirms neither shipped consumer imports a by-value global.

**Not verified:** the GUI itself was never launched — `cv2`, `ultralytics` and `torch` are not installed in
the environment where this was fixed.

### ~~26. Committed `.pkl` was pickled by an older scikit-learn~~ — RESOLVED 2026-08-19
`Data/pothole_ai_model.pkl` records `_sklearn_version = 1.7.2` (read directly from the pickle bytes at
offset 408, not inferred from the warning text). `pothole_detect_physics/requirements.txt` now pins
`scikit-learn==1.7.2`, with a comment explaining that retraining requires updating the pin.

Verified: 1.7.2 installs on Python 3.12 with **no numpy downgrade** (it needs only `numpy>=1.22.0`); the
model loads with `InconsistentVersionWarning` escalated to an exception and does **not** raise; the suite
passes 7/1 with identical numbers (event recall 1.00, precision 1.00), confirming the mismatch had not been
skewing results — it is simply no longer a risk.

### ~~25. Stage-1 recall metric is structurally misleading~~ — RESOLVED 2026-08-19
`test_stage1_accuracy_against_real_labels` now scores **per event**: contiguous `label==1` runs are grouped
by `group_label_events()`, and a pothole counts as detected if the cascade triggered anywhere inside its
block or within `EVENT_MATCH_TOLERANCE_S` (0.02 s) after it. Per-sample figures are still printed for
continuity with older baselines, labelled with their own ceiling and explicitly marked as not a quality
signal. The tolerance is converted to samples using the dataset's own timestep, so it survives a change of
sampling rate — relevant for CARLA runs that may not be 400 Hz.

Added `test_label_event_grouping`, which tests the grouping logic without loading the model, plus a
vacuity guard that fails if the analysed slice contains no labelled events.

Measured after the fix: **3 events, recall 1.00, precision 1.00** (per-sample recall still 0.08, ceiling
0.083). Metric discrimination was verified separately against synthetic scenarios — missed events, spurious
triggers and out-of-tolerance triggers all score as expected, so it can register a regression rather than
passing vacuously.

### ~~1. `model/road_seg.pt` is missing~~ — RESOLVED 2026-08-19
The real 7-class `yolo11s-seg` visible-road model was copied in from the upstream `pothole-detection-app`
repository, and `pothole_detection_app/.gitignore` was fixed so git actually tracks it. Two-stage detection
is now live everywhere. Model spec: [`11-vision-pipeline.md`](11-vision-pipeline.md). See also the new
issue #12b, which this resolution created.

---

## Maintaining this file

Rule 5: when you fix something here, **move it to the Resolved section** — do not leave a fixed issue in the
active list, and do not renumber the remaining ones. Record the fix in [`50-changelog.md`](50-changelog.md) with the issue number. When you discover something new, add it with a severity, a category, a concrete symptom, and a fix or workaround.
