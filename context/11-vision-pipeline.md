# 11 — Vision Pipeline (runtime)

**Scope:** `pothole_detection_app/app/` and its entry points — YOLO detection, two-stage road masking, both Tkinter GUIs, the shared utils.
**Training and dataset scripts are in [`12-vision-training-scripts.md`](12-vision-training-scripts.md).**
**Last verified against:** commit `3c586b8` (2026-08-19).

---

## The central idea: two-stage detection

A pothole detector trained on road photos will happily fire on anything dark and roughly elliptical — a shadow under a parked car, a patch of vegetation, a puddle on the pavement, a manhole in a driveway. The fix used here is to **constrain detections geometrically**: first work out which pixels are actually *visible drivable road*, then reject any detection whose centre is not on those pixels.

"Visible" is doing real work in that phrase. The segmentation classes are split into two groups:

- **Include** (road-like): `road, lane, street, asphalt, pavement, driveable, drivable`
- **Exclude** (occluders and artefacts): `car, truck, bus, van, vehicle, motorcycle, motorbike, bike, bicycle, person, pedestrian, rider, shadow, pole, barrier, curb, sidewalk, building, vegetation, tree, bush, grass, sky, obstacle, trash, bin, debris, cone, street light, traffic light, traffic sign, roadside object`

The road mask is `union(include classes) MINUS union(exclude classes)`. A pothole "seen" through a car is not seen.

> **Terminology collision:** this file's "stage 1 / stage 2" means *road segmentation / pothole detection*. The integration layer's "Stage 1 / Stage 2" means *sensor / vision*. They are unrelated. Say which you mean.

---

## The shipped road segmentation model — `model/road_seg.pt`

Added 2026-08-19 from the upstream `pothole-detection-app` repository. Metadata read directly from the
checkpoint (it is a zip; `best/data.pkl` holds the pickled config):

| Property | Value |
|---|---|
| Architecture | `yolo11s-seg.yaml` |
| Task | `segment` |
| Size | 20,507,364 bytes |
| SHA-256 | `b0330552c08c1945808e4f85f8ef5b3a0d75f6a25f0799b0198b493ab186c530` |
| Ultralytics version | 8.4.22 |
| Saved | 2026-03-18 |
| Trained on | `data/visible_road_seg_public_full/data.yaml` |
| Training args | 60 epochs, patience 30, batch 10, imgsz 640 |

**This is the real multi-class visible-road model, not the COCO stopgap** that
`scripts/download_road_model.py` fetches. It corresponds to the **`extended`** class preset in
`scripts/prepare_visible_road_public_dataset.py`:

| id | class |
|---|---|
| 0 | `visible_road` |
| 1 | `vehicle` |
| 2 | `pedestrian` |
| 3 | `shadow` |
| 4 | `vegetation` |
| 5 | `roadside_object` |
| 6 | `road_obstacle` |

### How these names resolve against the matching code

Both road-mask implementations use substring matching on class names, so it is worth knowing exactly
what they produce with this model. Traced by hand against the source; **not** verified by running.

**`TwoStageDetector` (`two_stage_detection.py`)** — include keywords match on the substring `road`:

- include ids resolve to `{0 visible_road, 5 roadside_object, 6 road_obstacle}` — note 5 and 6 match
  only because they contain the substring "road".
- exclude ids resolve to `{1, 2, 3, 4, 5, 6}` (`roadside object` matches the explicit
  `roadside object` keyword; `road obstacle` matches `obstacle`).
- The exclude pass runs **after** the include pass and subtracts, so 5 and 6 are added and then
  removed again. Net mask = `visible_road` minus all occluders. **Correct — but only because of the
  ordering.** If anyone reorders those two passes, the mask silently gains roadside clutter.

**`PotholeAppFiltered._get_drivable_class_ids()`** — matches road tokens then rejects on blocked
tokens (`object`, `obstacle`, …), so `roadside_object` and `road_obstacle` are filtered out by name
and drivable resolves cleanly to `{0}`. No ordering dependency here.

The filtered GUI's surrogate fallback looks up the literal tokens `visible_road` and
`roadside_object` — both exist in this model, so that path is now reachable.

---

## `app/two_stage_detection.py` — `TwoStageDetector`

**The reusable core of the vision subsystem.** This is what `integration/vision_adapter.py` and `scripts/predict_videos.py` both import. Changes here propagate everywhere — treat it as shared API.

### Construction

```python
create_two_stage_detector(pothole_model_path="model/best.pt",
                          road_model_path="model/road_seg.pt")
```

- Always loads the pothole model (`YOLO(pothole_model_path)`).
- Loads the road model only if the path is truthy **and** exists; on any load exception it logs a warning and continues.
- `self.use_road_seg` is the single flag that everything else branches on.

`model/road_seg.pt` **is present**, so `use_road_seg` is `True` and the segmentation half is live. If you ever see `Road segmentation model not found` in the logs, the file has gone missing and detection has silently dropped to unfiltered single-stage — historically the most common source of "why are there so many false positives".

### Class-name resolution

`_normalize_class_name()` lowercases and converts `-` and `_` to spaces, so `visible_road`, `Visible-Road`, and `visible road` all match. `_resolve_class_ids(names, keywords)` accepts YOLO `names` as either a dict or a list and returns the set of ids whose normalised name **contains** any keyword. Substring matching, so `road` also matches `roadside object` — which is why the exclude set is applied *after* and subtracts.

### `get_road_mask(frame, lowres_width=None)`

1. If `use_road_seg` is False → return an all-255 mask (everything is road; no filtering).
2. Optionally downscale to `lowres_width` (**800 px** in both GUI and video callers) for speed; the mask is upscaled back at the end with `INTER_NEAREST`.
3. Run the segmentation model.
4. **Include pass:** if any include-class ids resolved, OR together the masks of boxes with those class ids.
   **Fallback heuristic** (unknown labels): keep a mask if more than 30 % of its area lies below the top third of the frame — a crude "roads are at the bottom of the image" prior.
5. **Exclude pass:** OR together occluder masks and `AND NOT` them out of the road mask.
6. Morphological `CLOSE` then `OPEN` with a 5×5 kernel to fill holes and drop speckle.
7. **Empty-mask fallback:** if nothing survived, set the lower 60 % of the frame to road and log a warning.
8. On any exception: log and return an all-255 mask (fail open — never lose detections to a segmentation crash).

Masks are binarised at `> 0.5` then scaled to 0/255. Downstream tests use `> 127`.

### `detect_potholes(frame, conf=0.35, return_mask=False, road_mask=None, lowres_width=None)`

- Computes the road mask unless one is passed in (`road_mask=` lets video callers reuse a mask across frames).
- Runs the pothole model on the **full, unmasked frame** — deliberately. Feeding a blacked-out image to the detector destroys context and hurts accuracy; masking is applied to the *results*, not the *input*.
- **Hard filter:** for each box, clamp the centre into bounds and drop the box if `road_mask[cy, cx] <= 127`. Implemented by reassigning `results.boxes = results.boxes[keep_indices]`, so **every consumer sees the filtered set** — including `len(results.boxes)` and the confidence mean the integration layer computes. This is the key difference from merely hiding boxes at draw time.
- Returns `results`, or `(results, road_mask)` when `return_mask=True`.

### `visualize(frame, results, road_mask=None, show_mask=True)`

Green boxes with a filled label bar reading `Pothole {conf:.2f}`, a redundant per-box road check (defence in depth), a translucent green road overlay with drawn contours when `show_mask`, and a `Two-Stage Detection` / `Single-Stage Detection` banner top-left.

---

## `app/utils.py` — shared single-image helpers

Used by `main_enhanced.py` and `enhanced_utils.py`. **Not** used by `pothole_app_filtered.py`, which loads YOLO itself.

| Symbol | Purpose |
|---|---|
| `ROOT`, `MODEL_DIR`, `OUTPUT_DIR`, `DEFAULT_MODEL_PATH` | Path anchors derived from `__file__`. `OUTPUT_DIR` is created at import |
| `load_model(path=None)` | Loads and **caches** YOLO in the module global `_model`. Second call returns the cache and ignores the argument |
| `set_conf_threshold(v)` | Sets the module global `_conf` |
| `_imwrite_unicode(path, img)` | `cv2.imencode` + raw file write — works around OpenCV's failure on non-ASCII paths on Windows |
| `apply_roi_mask(img, roi)` | Two formats: `{"vertices": [(x,y)...]}` normalised polygon via `fillPoly`, or legacy `{left,right,top,bottom}` normalised rectangle that zeroes the outside |
| `run_detection(image_path, save_path=None, roi=None)` | Full single-image path; returns `(output_path, stats)` where stats has `num_potholes`, `confidences`, `avg_confidence`, `inference_time` (ms) |
| `pil_resize(path, max_size)` | Loads, `verify()`s, re-opens, converts RGB, thumbnails — for GUI display |
| `ensure_dirs()` | Creates `input/` and `output/` |

**Logging is configured at import of this module** — `logging.basicConfig` with a `FileHandler("pothole_detection.log")` and a stream handler. The log file lands in the **current working directory**, not a fixed path.

> **`get_conf_threshold()`** is the read-side counterpart to `set_conf_threshold()`, added 2026-08-19.
> **Always use it from other modules; never `from utils import _conf`.** That import binds the float by
> value at import time, so it stays 0.35 forever however many times `set_conf_threshold()` is called —
> which silently disconnected the enhanced GUI's confidence slider from detection. Fixed; both import
> blocks now carry a comment saying so. History in [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md).
> The same applies to `_model`: call `load_model()`, do not import the global.

---

## `app/enhanced_utils.py` — batch, preprocessing, monitoring

Re-imports the names above from `utils` (including the stale `_conf`).

- **`batch_process_images(folder, output_folder=None, save_csv=True, save_json=True, roi=None)`** — globs `.jpg/.jpeg/.png/.bmp/.webp` in both cases, runs the single-stage model per image (**no road segmentation in the batch path**), writes `det_<name>` annotated images plus `results.csv` and `results.json` into `output/batch_<timestamp>/`. Per-image exceptions are caught and recorded as an `error` row rather than aborting the run.
- **`preprocess_image(path, enhance=True, denoise=True, resize_max=None)`** — optional downscale, `fastNlMeansDenoisingColored(10,10,7,21)`, then PIL brightness ×1.1 and contrast ×1.2. The denoiser is **slow** (hundreds of ms per image); that is why it is opt-in.
- **`PerformanceMonitor`** — accumulates `(num_potholes, inference_time, confidence)`; `get_stats()` returns totals, min/avg/max inference time, FPS since construction, and mean confidence over non-zero entries.
- **`export_detections_csv/json(detections, path)`** — flat exports of the GUI's detection history.

---

## GUI 1 — `app/pothole_app_filtered.py` (`PotholeAppFiltered`)

Window title **"PAVE - Smart Pothole Event Review"**, 1400×900, min 1200×760. Launched by `run_app.py`. The newer and more capable of the two.

### What makes it different
Realtime video with **per-event artefacts on disk**: it does not just annotate, it produces a structured event record for every frame that contains a detection — the closest thing in the repo to a production detection log.

### Model loading
Loads `model/best.pt` and, if present, `model/road_seg.pt` directly with `YOLO(...)`. It does **not** use `utils.load_model`, so it has its own model instances and its own confidence variable (`self.conf_threshold`, a real `DoubleVar` — the `_conf` bug does not affect this app).

### Layout
- **Left panel:** file chooser · confidence slider (`DoubleVar`, default 0.35, live label) · "Use Road Mask" and "Show Road Overlay" checkboxes · scrollable class-filter list (mouse-wheel bound on enter/leave) with a *Detection Classes* group from the pothole model's `names` and a *Road Type Filter* group from the segmentation model's `names` · flagged-event listbox · Start / Stop / Save buttons.
- **Right panel:** KPI strip (`Video:`, `Frame:`, `Detections:`, `Flagged Saved:`) over the display canvas, and a status bar.

### Road mask — its own implementation
`get_road_mask()` here is **separate from and different to** `TwoStageDetector.get_road_mask()`. Do not assume they behave alike.

- Runs segmentation at `conf=0.5` on the **full-resolution** frame (no downscale).
- Keeps only classes in `enabled_road_classes` (a snapshot of the checkboxes intersected with `_get_drivable_class_ids()`).
- `_get_drivable_class_ids()` matches tokens `road, lane, drivable, street, asphalt, pavement` while blocking `object, obstacle, vehicle, pedestrian, shadow, vegetation`; if nothing matches it falls back to the first class id.
- **Surrogate fallback:** if the selected road classes yield less than 0.5 % of frame area but a `visible_road` class was selected and a `roadside_object` mask exists, it borrows the `roadside_object` mask as a stand-in. A workaround for mislabelled segmentation models.
- **`_refine_road_mask()`** — the distinctive part. Morphological close, then `connectedComponentsWithStats`, keeping only components that are all of: area ≥ `max(300, 0.1% of frame)`, bounding box reaching below 78 % of frame height, and centroid horizontally within 12–88 %. In other words: *the road is a large blob that touches the bottom of the frame near the centre.* If nothing qualifies, fall back to the lower half of the binary mask.

### Detection — `_detect_frame(frame, settings)`
Runs the pothole model on the full frame, then per box: skip if the class is unchecked; skip if the box centre is off-mask; otherwise count, draw (green for class 0, blue otherwise), and label with the class name. Draws the translucent road overlay if enabled, then a `Potholes: N` counter.

### Threading model
`_snapshot_runtime_settings()` copies every Tk variable into a plain dict **before** the worker starts — Tk variables are not safe to read from a non-main thread, and this is how the app avoids it. The worker (`_video_worker`) decodes and infers; `_consume_video_queue` runs on the Tk main loop via `root.after(15, ...)` and draws. The queue is `maxsize=2` and the worker **drops the oldest frame** when full, so display lags never stall inference. `WM_DELETE_WINDOW` is bound to `_on_close`, which stops the worker before destroying the window.

### Output artefacts — per video run
```
output/video_detect_<YYYYmmdd_HHMMSS>/
├── frames/<EVENT_ID>.jpg       raw frame
├── annotated/<EVENT_ID>.jpg    annotated frame
├── road_mask/<EVENT_ID>.png    mask, if one existed
└── metadata/
    ├── <EVENT_ID>.json         one record per event
    ├── events.jsonl            append-per-event
    └── events_summary.json     all records, written at end of run
```

Event id format: `PH-<YYYYmmddHHMMSS>-<frame_idx:06d>-<saved_seq:04d>`.

**GPS here is fake** — `_random_dummy_gps()` jitters ±0.0008 deg around a base that is itself randomised near **37.7749, -122.4194 (San Francisco)** at the start of each video. Note this differs from the integration layer's mock, which sits near **23.2599, 77.4126 (Bhopal)**. Neither is real.

Record shape is in [`20-data-contracts.md`](20-data-contracts.md). Each record carries a `confidence` — the mean over boxes surviving class and road-mask filtering (added 2026-08-20; `_detect_frame` returns them as a fourth value). **`integration/filtered_gui_adapter.py` publishes these to the map**, one-shot or `--watch` to follow a live run.

---

## GUI 2 — `app/main_enhanced.py` (`PotholeAppEnhanced`)

Window title **"PAVE"**, 1100×950. Launched by `quick_start.bat` and by the READMEs' `python app/main_enhanced.py`.

### Features not in the filtered app
- **Image / video mode toggle**, with the upload button relabelling itself.
- **Batch folder processing** via `enhanced_utils.batch_process_images`.
- **CSV / JSON export** of the in-session detection history.
- **Live performance monitor** panel.
- **Keyboard shortcuts** (`Ctrl+O` open, `Ctrl+S` save, etc.).
- **Model browser** — pick a different `.pt` at runtime.
- Optional **preprocessing** (enhance / denoise) before detection.
- Uses `TwoStageDetector` (so it benefits from road segmentation when the model exists), constructed in `__init__`.

### Video path
`_detect_video_worker` runs on a daemon thread and **writes an mp4** (`mp4v` fourcc) rather than only displaying. Key optimisation: the road mask is recomputed only every `ROAD_MASK_STRIDE = 10` frames and reused in between, at `ROAD_MASK_WIDTH = 800`. Progress is pushed to the Tk thread roughly ten times over the video via `_poll_video_queue` on a 50 ms `after` loop.

### Known rough edges
- ~~Subject to the `_conf` import trap~~ — fixed 2026-08-19; the slider now reaches detection.
- `_toggle_road_seg()` only refreshes the preview; its own comment admits it does not change the detector.
- If `results.boxes` is empty, `len(results.boxes)` is still used for stats — fine, but `avg_conf` is set to 0.0 by an explicit branch.

---

## `run_app.py` and `test_detection.py`

- **`run_app.py`** — prints root dir and whether `best.pt` / `road_seg.pt` exist, imports Tkinter and `PotholeAppFiltered` with progress prints, and wraps everything in a try/except that prints a full traceback. It is the friendliest launcher; use it when diagnosing import or model problems.
- **`test_detection.py`** — headless smoke test: load models, find a test image, build a road mask, run detection, report how many detections are on-road, save `output/test_result.jpg`. It looks for images in `input/` and `data/visible_road_seg_public_full/images/val/` — the latter is the **real training-dataset directory** for `road_seg.pt` (confirmed from the checkpoint's recorded `data.yaml` path), not a typo. Neither directory is in this repo, and it writes to `OUTPUT_DIR` without creating it, so expect it to exit early.

---

## Practical guidance

| Symptom | Look at |
|---|---|
| Too many false positives | Check the logs for `Road segmentation model not found` — if `road_seg.pt` went missing, nothing is being filtered. Otherwise raise `conf` |
| Detections vanish entirely | The road mask is too aggressive. Check `_refine_road_mask` component rules, or the include/exclude keyword sets against your model's actual `names` |
| Confidence slider does nothing (enhanced GUI) | Was the `_conf` import trap, fixed 2026-08-19. If it recurs, check nothing has gone back to `from utils import _conf` |
| Video processing is slow | Raise `ROAD_MASK_STRIDE`, lower `ROAD_MASK_WIDTH`, or disable road segmentation |
| Unicode path write failures | Use `_imwrite_unicode`, not `cv2.imwrite` |
| Need detection from other code | Import `create_two_stage_detector`, not the GUIs |
