# 02 — Repository Map

**Scope:** every file and directory in the repo, what it does, and whether it exists on a fresh clone.
**Last verified against:** commit `3c586b8` (2026-08-19).

Legend: **[C]** committed and present · **[G]** gitignored, absent on fresh clone · **[X]** referenced by code but does not exist

---

## Root

```
pothole-detection-full-pipeline/
├── CLAUDE.md                  [C] AI agent entry point — points here
├── README.md                  [C] Human-facing README. Rewritten 2026-08-20 to match reality
├── .gitignore                 [C] Root ignores; excludes secrets, venvs, pave_events.json
├── context/                   [C] This knowledge base
├── carla_sim/                 [C] CARLA testbed (Level A runs; Level B assets started)
├── integration/               [C] Cascade orchestrator — the glue layer
├── pothole_detect_physics/    [C] Sensor / physics / classical-ML subsystem
├── pothole_detection_app/     [C] YOLO vision subsystem
└── pothole_map_ui/            [C] PAVE map dashboard
```

**Root `.gitignore` excludes:** `venv/`, `__pycache__/`, `.pytest_cache/`, `integration/pave_events.json`, `pothole_detection_app/data/sample_images/*` (except its `README.md`), `**/Ultralytics/settings.json`, `.env`, `*.key`, `*.log`, `pothole_detection.log`, `pothole_map_ui/config.js`, IDE dirs.

---

## `carla_sim/` — CARLA simulation testbed

**Level A runs. Level B assets started 2026-09-08.** Design and rationale:
[`15-carla-testbed-plan.md`](15-carla-testbed-plan.md).

| File | | Purpose |
|---|---|---|
| `config.py` | [C] | Every tunable. Two values are `None` until `verify_setup.py` measures them |
| `verify_setup.py` | [C] | **P0.** IMU gravity convention, wheel position units, impulse API, tick rate |
| `scenario/potholes.py` | [C] | Pothole registry, route placement, wheel-over detection, ground-truth labels |
| `scenario/impulse.py` | [C] | Level A jerk model — seeds the drop, lets the suspension make freefall and impact |
| `scenario/route.py` | [C] | Deterministic route + minimal waypoint follower |
| `scenario/sensors.py` | [C] | Sensor rig: IMU/GNSS every tick, camera at 20 Hz |
| `scenario/tiles.py` | [C] | **Level B.** Spawns real pothole tiles on the road instead of scripting a jerk; `control=True` uses the flat tile |
| `scenario/drive_and_record.py` | [C] | Main loop; writes `sensors.csv` in contract #1. `--level A\|B`, `--control` |
| `scenario/export_map.py` | [C] | Exports the town's road network as GeoJSON for the dashboard's CARLA mode |
| `analyse_run.py` | [C] | Runs the sensor stage over a recording; reports detections **and the az envelope** |
| `README.md` | [C] | Setup, commands, gotchas |
| `assets/generate_pothole_meshes.py` | [C] | **Level B.** Parametric pothole tiles, runs headless in Blender; emits FBX + CARLA manifest |
| `assets/verify_pothole_meshes.py` | [C] | Re-imports each FBX and **measures** it — scale, bowl depth, and that collision leaves the bowl empty |
| `assets/fix_package_paths.py` | [C] | Repairs the asset paths CARLA writes into `PavePotholes.Package.json`; run after every import (issue #40) |
| `assets/probe_tile_collision.py` | [C] | Spawns each imported tile in a live simulator and ray-probes it — the test that proves the bowl is a real cavity, not a sealed hull |
| `assets/fbx/` | [C] | 4 generated FBX + `PavePotholes.json` (CARLA `Import.py` props format) |
| `out/` | [G] | Recorded runs |

Produces `sensors.csv` **column-identical to** `synthetic_pothole_dataset.csv`, which is what lets
the whole existing pipeline consume it unchanged.

---

## `integration/` — cascade orchestrator

Newest subsystem. Detail: [`14-integration-layer.md`](14-integration-layer.md).

| File | | Purpose |
|---|---|---|
| `schema.py` | [C] | `PotholeCandidateEvent` dataclass — the one shape every stage agrees on |
| `sensor_adapter.py` | [C] | `SensorSession` — wraps `PotholeDetector` + the `.pkl`; returns `ai_score` via `predict_proba` |
| `vision_adapter.py` | [C] | `VisionSession` — wraps `TwoStageDetector`; returns a mean-confidence score |
| `fusion.py` | [C] | Gate + weighted fusion. Holds `SENSOR_THRESHOLD`, `VISION_THRESHOLD`, weights |
| `frame_provider.py` | [C] | **Mocks** the frame↔sensor link and GPS. Searches several candidate image folders; `MockFrameProvider` returns `None` rather than raising when none are found |
| `carla_frame_provider.py` | [C] | **Real** timestamp-keyed frame + GNSS lookup over a recorded CARLA run. The thing the mock has been faking |
| `carla_replay.py` | [C] | **Added 2026-09-06.** Replays a recorded run onto the dashboard: publishes contract #12 (`vehicle_position.json`) and drops sensor-stage markers. Resolves #28 |
| `orchestrator.py` | [C] | `run(limit, dataset_path, frame_provider)` — main loop; argparse CLI with `--limit`, `--dataset`, `--carla-run` |
| `pave_connector.py` | [C] | `send_record_to_pave(record)` — the single writer for `pave_events.json`; `send_to_pave(event)` shapes cascade events and delegates |
| `filtered_gui_adapter.py` | [C] | Translates the filtered GUI's `events.jsonl` into contract #7 and publishes it. `--watch` follows a live run |
| `test_integration.py` | [C] | pytest — plumbing tests plus one real Stage-1 accuracy test |
| `pave_events.json` | [G] | Output. Created on first confirmed event. Polled by the map UI |

No `__init__.py` — `integration/` is **not** a package. Modules import each other by bare name, which works because running `python integration/orchestrator.py` puts `integration/` on `sys.path[0]`.

---

## `pothole_detect_physics/` — sensor subsystem

Detail: [`10-physics-sensor-pipeline.md`](10-physics-sensor-pipeline.md).

| File | | Purpose |
|---|---|---|
| `detector_py/pothole_detection.py` | [C] | `PotholeDetector` — the IMU finite state machine. **The core algorithm.** |
| `detector_py/generate_dataset.py` | [C] | Generates the synthetic 80,000-row CSV. Overwrites the committed one |
| `Model/features.py` | [C] | **Added 2026-09-06.** Shared feature construction: event ids, grouping key, rolling features. Imported by both scripts below so they cannot drift apart |
| `Model/train_ai_model.py` | [C] | Trains the RandomForest on a **grouped** (by event) split; `--rolling` adds the context features and writes a separate `.pkl` |
| `Model/run_detector_on_dataset.py` | [C] | Standalone demo: FSM + model over the CSV, prints metrics, plots. Inserts `detector_py/` on `sys.path` to import `PotholeDetector` |
| `Data/synthetic_pothole_dataset.csv` | [C] | 12.5 MB, 80,000 rows + header. Committed |
| `Data/pothole_ai_model.pkl` | [C] | 4.75 MB trained RandomForest. Committed |
| `requirements.txt` | [C] | pandas, numpy, scikit-learn, joblib, matplotlib |
| `README.md` | [C] | Subsystem README. Folder layout corrected 2026-08-19 |
| `.gitignore` | [C] | `__pycache__` only |

---

## `pothole_detection_app/` — vision subsystem

Detail: [`11-vision-pipeline.md`](11-vision-pipeline.md), [`12-vision-training-scripts.md`](12-vision-training-scripts.md).

### Runtime code — `app/`
| File | | Purpose |
|---|---|---|
| `app/two_stage_detection.py` | [C] | `TwoStageDetector` + `create_two_stage_detector()`. Road mask, hard-filtering, visualisation. **The reusable core** — this is what the integration layer imports |
| `app/pothole_app_filtered.py` | [C] | GUI **"PAVE - Smart Pothole Event Review"**. Realtime video, class filters, per-event JSON metadata. The newer, richer app |
| `app/main_enhanced.py` | [C] | GUI **"PAVE"** (enhanced). Batch processing, CSV/JSON export, perf monitor, keyboard shortcuts |
| `app/utils.py` | [C] | Model caching, global confidence, ROI masking, single-image detection, logging setup |
| `app/enhanced_utils.py` | [C] | Batch processing, image preprocessing, `PerformanceMonitor`, export helpers |

**Two GUIs, both alive.** `run_app.py` launches the *filtered* one; `quick_start.bat` and the READMEs launch the *enhanced* one. They do not share code.

### Entry points and config
| File | | Purpose |
|---|---|---|
| `run_app.py` | [C] | Launches `PotholeAppFiltered`; prints model-presence diagnostics first |
| `test_detection.py` | [C] | Headless smoke test of road-masked detection. Depends on absent dirs |
| `quick_start.bat` | [C] | Windows, 5 steps: organize (v2) → merge (v3) → train → evaluate → launch `main_enhanced.py` |
| `train_gpu.bat` | [C] | Windows: train medium/baseline |
| `requirements.txt` | [C] | ultralytics, opencv-python, pillow, numpy, torch, torchvision, lxml, tqdm |
| `data/data.yaml` | [C] | YOLO data config. **Contains a dead absolute path** |
| `LICENSE` | [C] | Proprietary, all rights reserved |
| `README.md` | [C] | 14.5 KB subsystem README |
| `.gitignore` | [C] | Excludes `*.pt` except `model/best.pt` and `model/road_seg.pt`, all datasets, `runs/`, `output/`, videos |

### Models — `model/`
| File | | Purpose |
|---|---|---|
| `model/best.pt` | [C] | 40.5 MB trained pothole detector. Single class: `pothole` |
| `model/road_seg.pt` | [C] | 20.5 MB visible-road segmentation model. `yolo11s-seg`, task `segment`, **7 classes** (`extended` preset). Enables the two-stage path everywhere. Details in [`11-vision-pipeline.md`](11-vision-pipeline.md) |

### Scripts — `scripts/`
| File | | Purpose |
|---|---|---|
| `train_model.py` | [C] | YOLO11 n/s/m training, three hyperparameter presets, auto-copies best to `model/best.pt` |
| `train_multiclass_road_seg.py` | [C] | Trains the multi-class visible-road segmentation model |
| `evaluate_model.py` | [C] | mAP@0.5, mAP@0.5:0.95, precision/recall, confusion matrix, multi-model comparison |
| `organize_dataset.py` | [C] | 70/15/15 split into `data/dataset_v2/`. **Dead hardcoded `SOURCE_DIR`** |
| `merge_datasets.py` | [C] | Merges VOC XML + existing YOLO sets into `data/dataset_v3/` |
| `prepare.py` | [C] | Older VOC XML to YOLO converter, writes `data/yolo/` |
| `prepare_multiclass_seg_dataset.py` | [C] | Builds a seg dataset from **CARLA** semantic renders |
| `prepare_visible_road_public_dataset.py` | [C] | 23 KB. Builds a seg dataset from Cityscapes / ACDC / IDD / Mapillary |
| `predict_videos.py` | [C] | CLI batch video inference with road segmentation |
| `predict_script.py` | [C] | Minimal single-image predict |
| `download_road_model.py` | [C] | Fetches `yolov8s-seg.pt`, copies it to `model/road_seg.pt` |
| `test_road_segmentation.py` | [C] | Sanity-checks segmentation output |

### Data and output directories — all absent on a fresh clone
| Path | | Notes |
|---|---|---|
| `data/raw/{images,annotations}/` | [G] | VOC XML source data |
| `data/dataset_v2/`, `data/dataset_v3/` | [G] | YOLO-format splits. `train_model.py` reads **v3**; `evaluate_model.py` reads **v2** |
| `data/visible_road_seg_public/` | [G] | Segmentation dataset used by `train_multiclass_road_seg.py` |
| `data/sample_images/` | [C] | Holds a tracked `README.md`; the photos themselves are gitignored. Preferred source of stand-in frames for the mock provider, with fallbacks to `input/` and the dataset splits |
| `input/`, `output/` | [G] | Runtime IO. `output/` is auto-created by `ensure_dirs()` |
| `training_results/`, `evaluation_results/`, `runs/` | [G] | Ultralytics artefacts |
| `venv/` | [G] | The `.bat` scripts assume it exists at `pothole_detection_app/venv/` |

---

## `pothole_map_ui/` — PAVE dashboard

Detail: [`13-map-ui.md`](13-map-ui.md).

| File | | Purpose |
|---|---|---|
| `index.html` | [C] | 3.3 KB. Structure: topbar, map, sidebar cards, history panel, toast. Loads `config.js` **before** `app.js` |
| `app.js` | [C] | 17.9 KB. Maps init, GPS tracking, Haversine proximity, pothole store, panels, `PotholeGuard` API, `pave_events.json` polling, dynamic Maps script loading |
| `styles.css` | [C] | 10.2 KB. Dark theme via `:root` custom properties |
| `config.example.js` | [C] | Template. Sets `window.GOOGLE_API_KEY` |
| `config.js` | [G] | **You must create this.** Copy the example, add a real key |

---

## Caveats about the READMEs

- **Root `README.md` — rewritten 2026-08-20, accurate.** Folder names, all five subsystems, the `config.js`
  key flow, an honest status table and the per-event accuracy figures. It deliberately does not mention
  `context/`, `CLAUDE.md` or `AGENTS.md`, because those are gitignored and will not exist for anyone who
  clones the repo. **Keep it that way.**
- **`pothole_detect_physics/README.md` — corrected 2026-08-20** for the location of
  `run_detector_on_dataset.py`.
- **`pothole_detection_app/README.md` — still stale.** It duplicates much of the old root README, including
  a model-performance table that presents upstream YOLO architecture ranges as though they were measurements
  of `best.pt`. Never quote those as measured (Rule 6).
