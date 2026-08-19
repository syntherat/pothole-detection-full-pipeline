# PAVE — Pothole Detection & Driver Alert System

An end-to-end pothole detection system built for the EPICS programme at Vellore Institute of
Technology, Bhopal. It detects potholes using **two independent sensing modalities**, fuses their
verdicts, and surfaces confirmed detections to a driver on a live map.

The design bet is that neither modality is reliable enough alone:

- A **camera** fails at night, in rain, when the road is occluded by the vehicle ahead, or when the
  pothole is full of water.
- An **IMU** only knows that something *felt* like a pothole. It cannot distinguish one from a speed
  breaker or a manhole cover, and it only ever sees potholes you have already driven into.

So they run as a **cascade**: a cheap sensor stage watches every sample, and only when it fires does
the expensive vision stage look at a camera frame. Most of the benefit of both, at a fraction of the
compute.

---

## Architecture

```
   pothole_detect_physics/                    pothole_detection_app/
   ┌────────────────────────────┐             ┌──────────────────────────────┐
   │ IMU stream (400 Hz)        │             │ Camera frame                 │
   │   ↓                        │             │   ↓                          │
   │ PotholeDetector (FSM)      │             │ road_seg.pt  → visible-road  │
   │   DROP → FREEFALL → IMPACT │             │                 mask         │
   │   → depth, length          │             │   ↓                          │
   │   ↓                        │             │ best.pt      → pothole boxes │
   │ RandomForest → ai_score    │             │   ↓                          │
   └───────────┬────────────────┘             │ drop boxes off the road      │
               │                              │   → vision_score             │
               │                              └───────────┬──────────────────┘
               │                                          │
               │        physics AND ai_score ≥ 0.50       │
               └──────────────► TRIGGER ──────────────────┘
                                   │
                        0.4·ai + 0.6·vision ≥ 0.50
                                   │
                                   ▼
                        integration/pave_events.json
                                   │
                          polled every 3 s
                                   ▼
                          pothole_map_ui/  (PAVE dashboard)
                     red markers · GPS tracking · 100 m proximity alert
```

Because vision carries 0.6 of the weight, a confirmed event **always requires the camera to see
something** — with `vision_score = 0` the maximum fused score is 0.4, below the 0.5 cut. The sensor
stage decides *when to look*; the vision stage decides *whether it counts*.

---

## Repository structure

```
pothole-detection-full-pipeline/
│
├── pothole_detect_physics/     Sensor subsystem — IMU physics + classical ML
│   ├── detector_py/
│   │   ├── pothole_detection.py    PotholeDetector — the state machine
│   │   └── generate_dataset.py     synthetic 80,000-row dataset generator
│   ├── Model/
│   │   ├── train_ai_model.py       RandomForest training
│   │   └── run_detector_on_dataset.py   standalone demo + plot
│   ├── Data/
│   │   ├── synthetic_pothole_dataset.csv   12.5 MB, committed
│   │   └── pothole_ai_model.pkl            4.75 MB, committed
│   └── requirements.txt
│
├── pothole_detection_app/      Vision subsystem — YOLO detection
│   ├── app/
│   │   ├── two_stage_detection.py  TwoStageDetector — the reusable core
│   │   ├── pothole_app_filtered.py GUI: realtime video + per-event metadata
│   │   ├── main_enhanced.py        GUI: batch, export, performance monitor
│   │   ├── utils.py                model cache, ROI masking, logging
│   │   └── enhanced_utils.py       batch processing, preprocessing, metrics
│   ├── scripts/                    dataset prep, training, evaluation, CLI inference
│   ├── model/
│   │   ├── best.pt                 40.5 MB — pothole detector, 1 class
│   │   └── road_seg.pt             20.5 MB — visible-road segmentation, 7 classes
│   ├── data/sample_images/         stand-in frames for the cascade (see its README)
│   ├── run_app.py                  launches the filtered GUI
│   ├── quick_start.bat             full Windows training pipeline
│   └── requirements.txt
│
├── integration/                Cascade — joins the three subsystems
│   ├── schema.py                   PotholeCandidateEvent, the shared event shape
│   ├── sensor_adapter.py           wraps the FSM + RandomForest
│   ├── vision_adapter.py           wraps TwoStageDetector
│   ├── fusion.py                   thresholds and weighted fusion
│   ├── orchestrator.py             the main loop
│   ├── frame_provider.py           mock frames + mock GPS (see Status)
│   ├── carla_frame_provider.py     real timestamp-keyed frames from a CARLA run
│   ├── filtered_gui_adapter.py     publishes GUI video events to the map
│   ├── pave_connector.py           writes pave_events.json
│   └── test_integration.py         pytest suite
│
├── pothole_map_ui/             PAVE dashboard — the driver-facing view
│   ├── index.html
│   ├── app.js                      maps, GPS, proximity, event polling
│   ├── styles.css
│   └── config.example.js           copy to config.js and add your Maps key
│
└── carla_sim/                  CARLA simulation testbed (scaffold — see Status)
    ├── verify_setup.py             P0 environment checks — run this first
    ├── config.py
    └── scenario/                   pothole registry, impulse model, recorder
```

---

## Status — what is real and what is simulated

Being precise about this matters more than it might seem, because several parts of the system look
finished and are not.

| Component | Status |
|---|---|
| Physics state machine | **Real**, working algorithm |
| RandomForest classifier | **Real**, trained — but on synthetic data only |
| Sensor dataset | **Synthetic.** Generated, not recorded. No real vehicle logs exist here |
| YOLO pothole detector (`best.pt`) | **Real** trained weights, committed |
| Road segmentation (`road_seg.pt`) | **Real** trained weights, 7-class, committed |
| Cascade, fusion, connector | **Real**, working |
| Map dashboard | **Real**, working. Uses genuine browser GPS |
| Frame ↔ sensor-row pairing | **Mocked.** The stand-in image has no relationship to the event |
| GPS in the cascade | **Mocked.** A straight line, not a route |
| Fusion weights | Hand-chosen, not fitted |
| CARLA testbed | **Scaffold only.** Written, never run against a simulator |

The single missing piece is **time-synchronised sensor and camera data from a real vehicle**.
Everything downstream of `frame_provider.py` is waiting on it — which is what the CARLA testbed is
being built to supply.

---

## Setup

Python 3.10+ (the code uses `X | None` unions). A CUDA GPU is optional for inference, recommended for
training.

```bash
python -m venv venv
```

Activate — Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source venv/bin/activate
```

Install both dependency sets (the cascade needs both):

```bash
pip install -r pothole_detect_physics/requirements.txt -r pothole_detection_app/requirements.txt pytest
```

> `scikit-learn` is **pinned to 1.7.2** — the version that pickled `pothole_ai_model.pkl`. Loading the
> model under a different version raises a warning that scikit-learn documents as possibly producing
> invalid results. If you retrain, update the pin.

Two extras are used by some scripts but not declared: `seaborn` for `scripts/evaluate_model.py`, and
`pyyaml` for `scripts/merge_datasets.py`.

---

## Running it

### Vision only — the quickest thing to see working

```bash
python pothole_detection_app/run_app.py
```

Opens the filtered GUI. Choose an image or video, adjust confidence, hit **Start Detection**. Video
runs write annotated frames and a per-event JSON log to `output/video_detect_<timestamp>/`.

The alternative GUI adds batch processing, CSV/JSON export and a performance monitor:

```bash
python pothole_detection_app/app/main_enhanced.py
```

### Sensor only

```bash
python pothole_detect_physics/Model/run_detector_on_dataset.py
```

Runs the state machine and the classifier across all 80,000 rows, prints a summary, and plots vertical
acceleration with detections marked.

### The full cascade

```bash
python integration/orchestrator.py --limit 2000
```

⚠ Without `--limit` this processes all 80,000 rows with a YOLO forward pass on every triggered one.

Add stand-in frames first — see `pothole_detection_app/data/sample_images/README.md`. Without them the
run still completes, but every triggered event is skipped for want of a frame.

### The map dashboard

```bash
cp pothole_map_ui/config.example.js pothole_map_ui/config.js
```

Add a Google Maps JavaScript API key to `config.js` (it is gitignored — **never commit it**), then
serve from the **repository root** so the dashboard can reach `integration/pave_events.json`:

```bash
python -m http.server 5500
```

Open `http://localhost:5500/pothole_map_ui/index.html`. Click **+ Simulate Detection** to see it work
with no backend at all.

### Live: GUI detections onto the map

```bash
python integration/filtered_gui_adapter.py pothole_detection_app/output/video_detect_<ts> --watch
```

Follows the GUI's event log while a video is processing and publishes each detection to the dashboard.

### Tests

```bash
pytest integration/test_integration.py -v -s
```

---

## Training

⚠ Training overwrites `model/best.pt`. Archive it first if you want to keep the current weights.

Full Windows pipeline — organize, merge, train, evaluate, launch:

```bash
pothole_detection_app\quick_start.bat
```

Or manually, from `pothole_detection_app/`:

```bash
python scripts/organize_dataset.py
```

```bash
python scripts/merge_datasets.py
```

```bash
python scripts/train_model.py --model small --hyperparams baseline
```

```bash
python scripts/evaluate_model.py --data data/dataset_v3/data.yaml
```

Model sizes are `nano` / `small` / `medium`; hyperparameter presets are `baseline` / `aggressive` /
`conservative`. Note that `organize_dataset.py` has a hardcoded source path you must edit first, and
that training reads `dataset_v3` while `evaluate_model.py` defaults to `dataset_v2` — pass `--data`
explicitly.

---

## Tuning

Nearly everything worth adjusting lives in `integration/fusion.py`:

| Constant | Default | Effect |
|---|---|---|
| `SENSOR_THRESHOLD` | 0.50 | How readily the sensor stage promotes to vision. **This is the recall ceiling of the whole system** — nothing downstream recovers a dropped event |
| `VISION_THRESHOLD` | 0.35 | YOLO confidence cutoff |
| `SENSOR_WEIGHT` / `VISION_WEIGHT` | 0.4 / 0.6 | Who decides. Setting `SENSOR_WEIGHT ≥ FUSION_THRESHOLD` destroys the camera's veto |
| `FUSION_THRESHOLD` | 0.50 | Final confirm/reject cut |

The state machine's own thresholds are constructor arguments on `PotholeDetector`
(`drop_margin`, `impact_margin`, `freefall_threshold`, `min_air_time`, `max_air_time`), so they can be
overridden per instance without editing the file.

`PROXIMITY_RADIUS_M` (100 m) and the dashboard theme tokens are at the top of `pothole_map_ui/app.js`
and `styles.css`.

---

## Accuracy — read this before quoting numbers

The only measurement this repository produces is **Stage-1 detection against the synthetic dataset**:

```
events in window : 3
detected         : 3
recall           : 1.00
precision        : 1.00
```

Scored **per event** — the state machine fires once per pothole while the dataset labels all 12 samples
of one, so per-sample recall caps at 1/12 regardless of detector quality. The test prints that figure
too, labelled, purely so older baselines stay comparable. Do not quote it.

**There is no vision-stage or end-to-end accuracy figure, and there cannot be one yet.** The frame
paired with each sensor event is an arbitrary stand-in, so `vision_score` and `final_confidence` from
such a run measure nothing. Any published YOLO benchmark ranges you may find are properties of the
architecture, not measurements of these weights.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Map shows "GOOGLE_API_KEY missing" | `config.js` not created — copy `config.example.js` |
| `Could not poll pave_events.json` every 3 s | Opened via `file://`, or not served from the repository root |
| GPS never locks | Geolocation needs `https://` or `localhost` |
| Warning: no stand-in images found | Add photos to `data/sample_images/` — the run still completes |
| `Road segmentation model not found` | `model/road_seg.pt` is missing; detection falls back to unfiltered single-stage |
| `ERROR: Data config not found` when training | `dataset_v3` does not exist — run `scripts/merge_datasets.py` |
| Many false positives | Check that road segmentation actually loaded; otherwise raise the confidence threshold |
| `ModuleNotFoundError: ultralytics` | Vision dependencies not installed |
| Tkinter missing on Linux | `apt install python3-tk` |

---

## Technology

**Sensor:** Python · NumPy · pandas · scikit-learn · joblib · matplotlib
**Vision:** Ultralytics YOLO11 · PyTorch · OpenCV · Tkinter · Pillow
**Dashboard:** HTML5 · CSS3 · vanilla ES6+ · Google Maps JavaScript API · Geolocation API — no build step
**Simulation:** CARLA 0.9.15 (scaffold)

## Acknowledgments

EPICS programme · Ultralytics for the YOLO implementation · pothole datasets from Kaggle and Roboflow ·
Cityscapes, ACDC, IDD and Mapillary for road segmentation data.

## License

Copyright © 2026. All rights reserved. This software is proprietary and confidential. Unauthorized
copying, transfer, modification, distribution or use, via any medium, is prohibited without prior
written permission from the copyright holder.

---

*Made for safer roads — combining sensor-based ML, computer vision, and real-time driver alerts.*
