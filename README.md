# PAVE: Pothole detection System

An end-to-end pothole detection and driver-alert system built for the EPICS project for Vellore Insitute of Technology, Bhopal. The system combines **two independent detection approaches** — a lightweight sensor-based machine learning classifier and a computer-vision YOLO model — with a **real-time in-car map dashboard** that visualizes detections and warns drivers when a pothole is nearby.

The hybrid design lets the system work in two complementary modes:

- **Sensor/Data-driven detection** — a classical ML model trained on road-condition style feature data (simulating accelerometer/vehicle-sensor input) to flag pothole vs. no-pothole road segments.
- **Vision-based detection** — a YOLOv11/YOLOv8 object detector that identifies potholes directly from road images and video frames, with an optional two-stage road-segmentation pass to cut down false positives.
- **Map UI** — a prototype dashboard that takes detections from either (or both) pipelines, plots them as live markers on Google Maps, tracks the vehicle's GPS position, and alerts the driver when a known pothole is within 100m.

Together these form a hybrid pipeline: detect potholes from sensor data and/or camera feed → log GPS-tagged detections → surface them to the driver in real time through the map dashboard.

---

##  System Architecture

```
                     ┌─────────────────────────┐
                     │   Sensor/ML Detector     │
                     │  (synthetic road-data     │
                     │   classifier, EPICS)      │
                     └────────────┬─────────────┘
                                  │
                                  ▼
┌───────────────────┐   detections   ┌─────────────────────┐
│  Vision Detector    │──────────────▶│    PAVE Map UI        │
│  (YOLOv11/v8, image/ │               │  (live dashboard,     │
│   video, two-stage)  │──────────────▶│   GPS + proximity      │
└───────────────────┘                │   alerts)              │
                                       └─────────────────────┘
```

Each detection pipeline can run independently, but both are designed to feed detections (with location metadata) into the PAVE dashboard for a unified driver-facing view.

---

##  Repository Structure

```
Hybrid-Pothole-Detection-System/
│
├── EPICS/                          # Sensor-based ML model (mathematical/classical ML)
│   ├── Data/
│   │   ├── pothole_ai_model.pkl
│   │   └── synthetic_pothole_dataset.csv
│   ├── detector_py/
│   │   ├── generate_dataset.py
│   │   ├── pothole_detection.py
│   │   └── run_detector_on_dataset.py
│   ├── Model/
│   │   └── train_ai_model.py
│   └── requirements.txt
│
├── pothole_detect_app/             # Vision-based YOLO detection app
│   ├── app/
│   │   ├── main_enhanced.py         # GUI application
│   │   ├── two_stage_detection.py   # Two-stage detector (road seg + pothole)
│   │   ├── utils.py
│   │   └── enhanced_utils.py
│   ├── scripts/
│   │   ├── organize_dataset.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── predict_script.py
│   │   ├── predict_videos.py
│   │   ├── download_road_model.py
│   │   └── test_road_segmentation.py
│   ├── data/
│   │   ├── data.yaml
│   │   ├── raw/
│   │   └── dataset_v*/
│   ├── model/                       # Trained YOLO weights (.pt)
│   ├── quick_start.bat
│   ├── train_gpu.bat
│   └── requirements.txt
|   ├── run_app.py
│   ├── test_detection.py
│   └── requirements.txt
│
├── pothole-map-ui/                 # PAVE — in-car map dashboard (prototype)
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── README.md                       # (this file)
```

---

## 1. Sensor-Based Detector 

A classical machine learning model developed as part of the EPICS program. It uses a **synthetic dataset** representing road surface/sensor conditions and trains a classifier to predict whether a road segment contains a pothole simulating how in-vehicle sensor data could be used for automatic pothole detection.

**Folder:** `pothole-detect-physics`

| Folder | Purpose |
|---|---|
| `Data/` | Stores the generated dataset and trained model (`.pkl`) |
| `detector_py/` | Dataset generation and detection scripts |
| `Model/` | Model training script |

### Requirements
```bash
pip install -r requirements.txt
```
Core libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`

### Usage

**Step 1 — Generate the dataset**
```bash
python detector_py/generate_dataset.py
```
Produces `Data/synthetic_pothole_dataset.csv`.

**Step 2 — Train the model**
```bash
python Model/train_ai_model.py
```
Saves the trained model to `Data/pothole_ai_model.pkl`.

**Step 3 — Run detection**
```bash
python detector_py/run_detector_on_dataset.py
```
Loads the trained model and runs pothole detection on the dataset.

### Technologies Used
Python · Pandas · NumPy · Scikit-Learn · Joblib



---

## 2️. Vision-Based Detector (YOLO Model)

A real-time computer-vision pothole detector powered by **YOLOv11/YOLOv8**, with a GUI, batch processing, and optional two-stage detection (road segmentation + pothole detection) to reduce false positives.

**Folder:** `pothole_detection_app/`

### Features
- **State-of-the-art models:** YOLOv11 (nano/small/medium) and YOLOv8
- **Real-time performance:** 2–15ms inference depending on model size
- **Adjustable confidence threshold:** 10–90%
- **Batch processing:** entire folders of images/videos
- **Two-stage detection:** road segmentation to cut false positives
- **Interactive Tkinter GUI:** live bounding boxes, confidence scores, stats
- **Full training pipeline:** dataset organization, training presets, evaluation (mAP, precision, recall)

### Requirements
- Python 3.8+ (3.10 recommended)
- CUDA-capable GPU (optional, recommended for training)
- 4GB+ RAM for inference, 8GB+ for training

```bash
git clone https://github.com/syntherat/pothole-detection-app.git
cd pothole_detect_app
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Place a trained model at `model/best.pt`, load one via the GUI, or train your own (below).

### Usage

**GUI application**
```bash
python app/main_enhanced.py
```
Upload an image → adjust confidence threshold (default 35%) → click **Detect Potholes** → results saved to `app/output/`.

**Batch image processing**
```bash
python scripts/predict_script.py --input ./input --output ./output --conf 0.35
```

**Video processing**
```bash
python scripts/predict_videos.py
```
Processes all videos in `input/`, saves results to `output/videos/`.

### Training Your Own Model

**Quick start (Windows)**
```bash
quick_start.bat
```
Organizes the dataset, trains a YOLOv11-small model with baseline hyperparameters, evaluates performance, and generates prediction examples.

**Manual training**

1. Place annotated images in `data/raw/images/` and `data/raw/annotations/` (VOC XML format), then:
```bash
python scripts/organize_dataset.py
```
Creates a 70/15/15 train/val/test split in YOLO format.

2. Train:
```bash
python scripts/train_model.py --model small --hyperparams baseline
python scripts/train_model.py --all                                   # train all sizes
python scripts/train_model.py --model medium --hyperparams aggressive --epochs 150
```

| Model | Speed | Use case |
|---|---|---|
| nano | ~2ms | Embedded systems |
| small | ~5ms | Balanced, recommended |
| medium | ~15ms | Highest accuracy, server deployment |

Hyperparameter presets: `baseline`, `aggressive` (heavy augmentation), `conservative` (light augmentation, small datasets).

3. Evaluate:
```bash
python scripts/evaluate_model.py
```
Generates mAP@0.5, mAP@0.5:0.95, precision/recall curves, confusion matrix.

### Two-Stage Detection (Advanced)

Reduces false positives by first segmenting the road surface, then detecting potholes only within it.

```bash
python scripts/download_road_model.py     # download road segmentation model
python scripts/test_road_segmentation.py  # verify segmentation quality
```

```python
from app.two_stage_detection import create_two_stage_detector

detector = create_two_stage_detector(
    pothole_model_path="model/best.pt",
    road_model_path="model/road_seg.pt"
)

results = detector.detect_potholes(image, conf=0.35)
annotated = detector.visualize(image, results, show_mask=True)
```

### Model Performance

Typical ranges — actual results depend on dataset size, quality, and diversity:

| Model | Size | Speed (ms) | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|---|
| YOLOv11n | 2.6MB | 1–2 | 55–70% | 60–75% | 50–65% |
| YOLOv11s | 9.4MB | 2–4 | 60–75% | 65–80% | 55–70% |
| YOLOv11m | 20MB | 5–8 | 65–80% | 70–85% | 60–75% |

*Benchmarked on NVIDIA RTX 4060. Well-annotated datasets with 2000+ diverse examples typically achieve the higher end of these ranges.*

### Dataset Format

**VOC XML (raw data):**
```xml
<annotation>
  <object>
    <name>pothole</name>
    <bndbox>
      <xmin>100</xmin><ymin>150</ymin><xmax>200</xmax><ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

**YOLO format (auto-converted):**
```
0 0.425 0.512 0.156 0.178
```
`class_id center_x center_y width height` (normalized 0–1)

### Configuration Notes
- **Confidence threshold:** lower (0.10–0.35) = more detections, more false positives; higher (0.60–0.90) = fewer, higher-confidence detections. Recommended start: 0.35–0.50.
- **Road ROI mask:** ignore roadside areas (trees, bushes) in GUI and video runs; ratios are 0–1 rectangles (left/right/top/bottom).
- **Custom model:** use "Select Model .pt" in the GUI, or `load_model("/path/to/your/model.pt")` in code.

### Logging & Output
Results are saved to `app/output/` as `pred_YYYYMMDD_HHMMSS_imagename.jpg`. Detection statistics are logged to `pothole_detection.log` (timestamp, log level, operation details).

### Troubleshooting

| Issue | Fix |
|---|---|
| "Model not found" | Ensure `model/best.pt` exists, or load one via "Select Model .pt" |
| "Failed to read image" | Check format (JPG/PNG/BMP) or file corruption |
| Slow detection | Enable GPU mode, reduce image resolution |
| Import errors | `pip install --upgrade -r requirements.txt`, use a virtual environment |

### Technology Stack
Ultralytics YOLOv11/v8 · PyTorch · OpenCV · Tkinter · Pillow · NumPy · lxml · tqdm

---

## 3️. PAVE Map UI (Prototype Dashboard)

A real-time, in-car dashboard prototype that visualizes detections from the pothole detection pipelines above. Plots detected potholes as red dots on a live Google Maps view, tracks the vehicle's GPS position, and alerts the driver when a pothole is within **100m**.

**Folder:** `pothole-map-ui/`

```
pothole-map-ui/
├── index.html    — HTML structure and layout
├── styles.css    — Styling, theme, and animations
└── app.js        — Logic: maps, GPS, proximity, detection handling
```

### What it does
- **Top bar:** logo, monitoring status pill, GPS status pill, "Potholes Detected" counter, Simulate button
- **Main view:** live Google Maps with the car's position, red markers for detected potholes
- **Sidebar:** boolean detection status card, GPS coordinates card, proximity alert card (pulses red when a pothole is nearby), and a live event feed
- **History panel:** full-screen overlay with a split list + map view of all past detections
- **Toast notifications** for new detections

### Theme
Dark, dashboard-style UI defined via CSS custom properties for easy re-theming:

```css
--bg          /* main background       #0a0c10 */
--surface     /* card/bar background   #111318 */
--surface2    /* inner card background #181c24 */
--border      /* border color          #1e2430 */
--accent      /* amber highlight       #f0a500 */
--danger      /* red alert color       #ff3b3b */
--safe        /* green safe color      #00e676 */
--font-head   /* Rajdhani  — display font */
--font-mono   /* JetBrains Mono — data/code font */
```

### Core Logic (`app.js`)

Configuration at the top of the file:
```js
const GOOGLE_API_KEY    = "YOUR_GOOGLE_MAPS_API_KEY"; // replace this
const MAP_CENTER        = { lat: 23.2599, lng: 77.4126 }; // default map center
const PROXIMITY_RADIUS_M = 100; // alert radius in meters
```

| Function | Description |
|---|---|
| `initMaps()` | Initializes main + panel maps with dark styling |
| `startLocationTracking()` | Starts `watchPosition` for continuous GPS tracking |
| `updateCarMarker(lat, lng, heading)` | Moves the car icon, rotates by heading |
| `updateGPSDisplay(lat, lng, accuracy)` | Updates GPS card, triggers proximity check |
| `checkProximity(userLat, userLng)` | Runs Haversine formula against stored potholes |
| `getDistanceMeters(lat1, lng1, lat2, lng2)` | Haversine distance calculation |
| `addPothole(lat, lng, locationName, detectedBy)` | Adds a detection to the store and UI |
| `placeRedDot(map, arr, ph, animate)` | Places a marker with an info window |
| `updateStatus(ph)` | Sets boolean status card TRUE→FALSE after 4s |
| `addToList(ph)` | Prepends detection to the live feed (max 10) |
| `updateBadge()` | Updates the detection count badge |
| `showToast(msg)` | Shows a 3.5s slide-up notification |
| `openPotholePanel()` / `closePotholePanel()` | Opens/closes the history panel |
| `simulateDetection()` | Drops a random test pothole within ~300m for testing |
| `getUserPosition()` | Returns car position, or falls back to `MAP_CENTER` |

### Setup

1. **Enable Maps JavaScript API** — in [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Library → enable *Maps JavaScript API*, and make sure billing is active.
2. **Add your API key** — in `app.js`, replace:
   ```js
   const GOOGLE_API_KEY = "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
   ```
3. **Restrict your key** (for public repos) — Credentials → your key → HTTP Referrers → add:
   ```
   localhost/*
   localhost:5500/*
   127.0.0.1/*
   ```
4. **Run** — open `index.html` in a browser, or serve via VS Code Live Server / any local HTTP server.

### Troubleshooting

| Problem | Fix |
|---|---|
| Map shows dark background, no tiles | Check browser console (F12) for the exact error |
| `ApiNotActivatedMapError` | Enable Maps JavaScript API in Google Cloud Console |
| `RefererNotAllowedMapError` | Add `localhost/*` to allowed HTTP referrers |
| `InvalidKeyMapError` | Re-copy the key, check for stray spaces |
| `styles.css 404` | Ensure all 3 files are in the same folder |
| Map loads but no red dots | Click "+ Simulate Detection", or call `window.PotholeGuard.reportDetection()` in console |
| GPS not locking | Allow location permission; use HTTPS or localhost |
| Proximity card not activating | GPS must be active and a pothole within 100m; increase `PROXIMITY_RADIUS_M` for testing |

### Tech Stack
| Layer | Technology |
|---|---|
| Structure | HTML5 |
| Styling | CSS3 with custom properties |
| Logic | Vanilla JavaScript (ES6+) |
| Maps | Google Maps JavaScript API |
| Location | Browser Geolocation API (`watchPosition`) |
| Fonts | Rajdhani + JetBrains Mono (Google Fonts) |
| Build | None — plain files, no bundler |

---

##  How the Pieces Fit Together

1. **Detection** happens via the sensor-based EPICS classifier and/or the YOLO vision model, either offline on datasets/video or (in the roadmap) in real time from a live camera/sensor feed.
2. Each detection is tagged with a **GPS location**.
3. Detections are pushed to **PAVE**, which plots them as red markers, tracks the driver's live position, and raises a proximity alert when the vehicle is within 100m of a known pothole.

This hybrid approach means the system isn't dependent on a single sensing modality — the sensor-based model can catch potholes when a camera view is poor, while the vision model adds precise, image-verified localization, and PAVE ties both into a single driver-facing safety layer.

---


##  Combined Technology Stack

**Sensor/ML model:** Python, Pandas, NumPy, Scikit-Learn, Joblib
**Vision model:** Ultralytics YOLOv11/v8, PyTorch, OpenCV, Tkinter, Pillow, NumPy
**Map UI:** HTML5, CSS3, Vanilla JavaScript, Google Maps JavaScript API, Browser Geolocation API

##  Acknowledgments

- EPICS program
- Ultralytics for the YOLO implementation
- Pothole datasets from Kaggle and Roboflow
- Open-source computer vision community

##  License

Copyright © 2026. All Rights Reserved.
This software is proprietary and confidential. Unauthorized copying, transfer, modification, distribution, or use of this software, via any medium, is strictly prohibited without prior written permission from the copyright holder.

---

*Made for safer roads — combining sensor-based ML, computer vision, and real-time driver alerts.*