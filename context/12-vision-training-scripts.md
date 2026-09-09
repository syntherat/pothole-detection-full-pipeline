# 12 — Vision Training & Dataset Scripts

**Scope:** `pothole_detection_app/scripts/` — dataset preparation, training, evaluation, batch inference.
**Rule 8 applies to everything in this file: training and dataset generation need explicit permission before you run them.**
**Last verified against:** commit `3c586b8` (2026-08-19).

---

## Two model families

The subsystem trains two different kinds of model, and the scripts are split accordingly.

| Family | Task | Script | Output |
|---|---|---|---|
| **Pothole detector** | Object detection, 1 class (`pothole`) | `train_model.py` | `model/best.pt` |
| **Road segmentation** | Instance segmentation, 4 or 7 classes | `train_multiclass_road_seg.py` | `model/road_seg.pt` |

**Both sets of weights are now committed.** `model/road_seg.pt` is a real 7-class `yolo11s-seg` model
trained on the `extended` preset — spec in [`11-vision-pipeline.md`](11-vision-pipeline.md). You only need the
scripts below to retrain or improve them.

---

## Dataset directory versions — mind the mismatch

There are three generations of pothole dataset directory, and **the scripts do not agree on which to use**:

| Directory | Written by | Read by |
|---|---|---|
| `data/yolo/` | `prepare.py` (oldest) | `data/data.yaml` |
| `data/dataset_v2/` | `organize_dataset.py` | **`evaluate_model.py`** |
| `data/dataset_v3/` | `merge_datasets.py` | **`train_model.py`** |

`quick_start.bat` now runs `merge_datasets.py` between the two, so the chain is organize (v2) then merge (v3) then train (v3) then evaluate (v3, passed explicitly). **If you invoke the scripts by hand you must still run `merge_datasets.py` yourself**, or training fails with "Data config not found". History in [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md) issue #4.

---

## Pothole detector training — `train_model.py`

### CLI
```
--model {nano,small,medium}                default: small
--hyperparams {baseline,aggressive,conservative}   default: baseline
--resume                                   resume from last checkpoint
--device <cuda-index|cpu>                  default: auto-detect
--all                                      train all three sizes with baseline
```

### Base weights
| Preset | Checkpoint |
|---|---|
| nano | `yolo11n.pt` |
| small | `yolo11s.pt` |
| medium | `yolo11m.pt` |

Downloaded by Ultralytics on first use.

### Hyperparameter presets
| | baseline | aggressive | conservative |
|---|---|---|---|
| epochs | 100 | 150 | 80 |
| batch | 16 | 16 | 16 |
| imgsz | 640 | 640 | 640 |
| lr0 | 0.01 | 0.02 | 0.005 |
| lrf | 0.01 | 0.001 | 0.01 |
| patience | 20 | 30 | 15 |
| mosaic | 1.0 | 1.0 | 1.0 |
| mixup | 0.0 | 0.1 | 0.0 |

Use **conservative** for small datasets (less augmentation, gentler LR, earlier stop), **aggressive** when you have plenty of diverse data.

### Fixed training arguments (same for all presets)
- Optimiser **AdamW**, momentum 0.937, weight decay 0.0005
- Warmup: 3 epochs, momentum 0.8, bias lr 0.1
- Augmentation: `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, `translate 0.1`, `scale 0.5`, `fliplr 0.5`
- **`flipud 0.0`, `degrees 0.0`, `perspective 0.0`** — deliberately no vertical flip or rotation. Road scenes have a fixed gravity orientation; flipping them vertically teaches nothing real.
- Loss weights: `box 7.5`, `cls 0.5`, `dfl 1.5`
- `seed 42`, `deterministic True`, `workers 8`, `save_period 10`, `cache False`

### Outputs
Runs land in `training_results/pothole_<size>_<preset>_<timestamp>/`. On success the best checkpoint is copied twice:
1. `model/best_<size>_<timestamp>.pt` (archived)
2. `model/best.pt` (**overwrites the live model**)

That second copy means a training run silently replaces the weights every GUI and the integration layer load. Archive the current `best.pt` before training if you care about it.

---

## Road segmentation training — `train_multiclass_road_seg.py`

Expects `data/visible_road_seg_public/data.yaml`. Presets map to `yolo11{n,s,m}-seg.pt`. Reads class names out of the data.yaml with a small hand-rolled parser (`read_names_from_data_yaml`) rather than pulling in PyYAML. `default_workers()` returns **0 on Windows** — Ultralytics' dataloader workers are unreliable there.

Produces the `road_seg.pt` that unlocks the entire two-stage path. **The committed model was built this
way** — `yolo11s-seg`, 60 epochs, batch 10, imgsz 640, on a dataset named `visible_road_seg_public_full`.
Note the committed model uses the **`extended`** 7-class preset, while this script's docstring lists only
the 4 `core` classes; the docstring is out of date relative to the shipped weights.

---

## Segmentation dataset preparation

### `prepare_visible_road_public_dataset.py` (23 KB — the substantial one)

Builds a unified YOLO-seg dataset from public driving datasets:

| Source | Label format |
|---|---|
| Cityscapes | `gtFine` labelIds + `leftImg8bit` |
| ACDC | `gt_labelIds` + `rgb_anon` |
| IDD Segmentation | `gtFine` polygons + `leftImg8bit` |
| Mapillary Vistas v1.2 / v2.0 | indexed label PNG + config |

**Class presets:**
- `core`: `visible_road, vehicle, pedestrian, shadow`
- `extended`: core + `vegetation, roadside_object, road_obstacle`

Cityscapes id mappings are hardcoded (road `{7}`, vehicles `{26..33}`, pedestrians `{24,25}`, vegetation `{21,22}`). Uses `ProcessPoolExecutor` for parallel conversion.

### `prepare_multiclass_seg_dataset.py`

Same output shape, but from **CARLA simulator** semantic renders. Expects `<input>/rgb/` and `<input>/semantic/` (single-channel CARLA class ids). CARLA tags hardcoded: road `7`, pedestrian `4`, vehicle `10`. Classes: `visible_road, vehicle, pedestrian, shadow`. Converts masks to YOLO-seg polygons via `findContours` with a minimum-area filter.

---

## Pothole dataset preparation

### `organize_dataset.py`
70/15/15 train/val/test split, `random.seed(42)`. Source images and YOLO `.txt` labels must sit side by side; pairs with empty label files are skipped. Writes `data/dataset_v2/{split}/{images,labels}/`.

⚠ **`SOURCE_DIR` is hardcoded** to `C:\Users\palso\OneDrive\Desktop\VITB\epics\Pothole Dataset` — a dead machine-specific path. Edit it or the script does nothing.

### `merge_datasets.py`
Merges a **VOC XML** set (`data/raw/images` + `data/raw/annotations`, ~665 images per its docstring) with an existing **YOLO** set (`data/dataset_v2`, ~1243 images) into `data/dataset_v3` (~1908 total). Contains `voc_to_yolo()`, which normalises and clamps boxes to `[0,1]` and assigns class `0`. Writes the `data.yaml` that `train_model.py` reads.

### `prepare.py`
The oldest converter. VOC XML → YOLO with `lxml`, single class map `{"pothole": 0}`, unknown classes skipped, writes `data/yolo/{images,labels}/{train,val}`. Superseded by `merge_datasets.py`; kept for reference.

### `data/data.yaml`
```yaml
path: C:\Users\palso\OneDrive\Desktop\VITB\epics\pothole_detect_app\data\yolo
train: images/train
val: images/val
names: [pothole]
```
⚠ Dead absolute path. The generated per-dataset `data.yaml` files are the ones actually used by training.

---

## Evaluation — `evaluate_model.py`

```
--model <path>      default: model/best.pt
--data <path>       default: data/dataset_v2/data.yaml
--conf <float>      default: 0.25
--compare <p1> <p2> ...   compare multiple models
```

Produces mAP@0.5, mAP@0.5:0.95, precision, recall, PR curves and a confusion matrix into `evaluation_results/eval_<model_stem>/`. ⚠ It still defaults to **dataset_v2** while training defaults to **v3** — `quick_start.bat` passes `--data` explicitly, and you should too when running it by hand.

Requires `matplotlib` and `seaborn`, neither of which is in `pothole_detection_app/requirements.txt`.

---

## Inference scripts

### `predict_videos.py`
The CLI counterpart to the GUIs' video mode. Imports `create_two_stage_detector` from `app/` via `sys.path` insertion.

```
--vids-dir <dir>        default: <repo_parent>/vids     ⚠ outside the repo
--model <path>          default: model/best.pt
--road-model <path>     default: model/road_seg.pt
--output-dir <dir>      default: output/videos
--conf <float>          default: 0.35
--vid-stride <int>      default: 1        process every Nth frame
--stream / --no-stream          default: on
--use-road-seg / --no-use-road-seg   default: on
--show-road-mask / --no-show-road-mask   default: off
```

Same performance strategy as the enhanced GUI: `ROAD_MASK_STRIDE = 10`, `ROAD_MASK_WIDTH = 800`. Writes annotated videos plus a CSV summary.

### `predict_script.py`
Sixteen lines. `python scripts/predict_script.py <image_path>`, uses the relative path `model/best.pt` so it **must be run from `pothole_detection_app/`**. Ultralytics chooses the save directory.

### `download_road_model.py`
Downloads `yolov8s-seg.pt` and installs it as a **last-resort stand-in** for `model/road_seg.pt`.

```
--output <path>   write elsewhere (default: model/road_seg.pt)
--force           overwrite an existing file (also writes a .bak first)
```

**It refuses to overwrite an existing model** and exits 1 unless you pass `--force` — added 2026-08-19,
because the repo now ships a real 7-class visible-road model and silently replacing it would be very hard to
notice. The check runs before the download, so a refused run costs nothing.

**Why it is only ever a stopgap:** the weights are **COCO-trained**, and COCO has **no road class**.
`TwoStageDetector`'s include-keyword resolution finds nothing and falls through to its "lower two-thirds of
the frame" heuristic. It makes the two-stage path *run*; it does not make it *work*. For a real model use
`prepare_visible_road_public_dataset.py` then `train_multiclass_road_seg.py`.

### `test_road_segmentation.py`
Sanity-checks segmentation output quality before you rely on it.

---

## Dataset formats

**VOC XML input:**
```xml
<annotation>
  <size><width>1280</width><height>720</height></size>
  <object>
    <name>pothole</name>
    <bndbox><xmin>100</xmin><ymin>150</ymin><xmax>200</xmax><ymax>250</ymax></bndbox>
  </object>
</annotation>
```

**YOLO detection label** — one line per object, all values normalised 0–1:
```
0 0.425 0.512 0.156 0.178
class_id center_x center_y width height
```

**YOLO segmentation label** — polygon points, normalised:
```
0 x1 y1 x2 y2 x3 y3 ...
```

---

## Recommended order for a full retrain

1. Assemble raw data into `data/raw/` (VOC XML) and/or a YOLO-format folder.
2. `python scripts/organize_dataset.py` — after fixing `SOURCE_DIR`.
3. `python scripts/merge_datasets.py` — produces the `dataset_v3` that training expects.
4. `python scripts/train_model.py --model small --hyperparams baseline` — **ask first (Rule 8)**; archive the existing `model/best.pt`.
5. `python scripts/evaluate_model.py --data data/dataset_v3/data.yaml`.
6. For the road model: `prepare_visible_road_public_dataset.py` → `train_multiclass_road_seg.py`.
