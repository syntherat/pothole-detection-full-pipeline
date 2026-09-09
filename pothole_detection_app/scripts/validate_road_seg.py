# scripts/validate_road_seg.py
"""
Validate road_seg.pt on its OWN training-distribution validation split.

This is the diagnostic that splits issue #30 in two. `road_seg.pt` never predicts
`visible_road` on dash-camera frames (0/16) or on CARLA frames (0/45), so the road
mask is always empty and the cascade always falls through to the lower-60% crop.
Two very different faults produce that same symptom:

  A. DOMAIN GAP    -- the checkpoint fires `visible_road` normally on the data it was
                      trained on, and simply does not transfer to dash-cam geometry.
                      Fix = fine-tune or keep the geometric prior as the real Stage 1.
  B. LOAD/PREPROC  -- it does not fire even in-distribution, so the fault is on our
                      side: wrong imgsz, wrong preprocessing, or a confidence gate the
                      model's own score distribution never clears.
                      Fix = align inference with the recorded training arguments.

The script reports three things, because "no visible_road" has more than one cause:

  [1] val-mode metrics, PER CLASS. The `visible_road` row is the answer to A vs B.
      Aggregate mAP hides it -- the checkpoint records 7-class aggregates only.
  [2] predict-mode sweep over the same val images at several confidence thresholds.
      `.val()` scores at conf=0.001; the app calls `road_model(frame)` with no conf
      at all, so ultralytics' predict default of 0.25 applies. A model can pass [1]
      and still emit nothing at [2] -- that gap IS fault B, and it is invisible to
      any mAP number.
  [3] how often TwoStageDetector.get_road_mask() falls through to the lower-60% crop
      on in-distribution images. This is issue #30's actual symptom, measured where
      the model should be at its best.

Reads the checkpoint's recorded train_args and evaluates at the same imgsz, so a
mismatch cannot be introduced by this script's own defaults.

Usage:
    python scripts/validate_road_seg.py
    python scripts/validate_road_seg.py --data D:/epics/pothole_detect_app/data/visible_road_seg_public_full/data.yaml
    python scripts/validate_road_seg.py --max-predict 200 --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DEFAULT_DATASET = ROOT / "data" / "visible_road_seg_public_full"

# The class the road mask is built from. Everything else road_seg.pt predicts is
# either excluded outright or excluded after inclusion -- see issue #30's class table.
ROAD_CLASS_NAME = "visible_road"

# Thresholds worth separating: 0.001 is what .val() uses, 0.25 is ultralytics'
# predict default and therefore what the app actually runs at, 0.05 is the floor
# issue #30 was measured at.
PREDICT_CONFS = (0.001, 0.05, 0.25)


def read_checkpoint_args(model_path: Path) -> dict:
    """Recover the training arguments recorded inside the checkpoint.

    Loaded with weights_only=False because ultralytics checkpoints hold pickled
    model objects; this file is repo-local and trusted.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    return dict(ckpt.get("train_args") or ckpt.get("model", {}).yaml.get("args", {}) or {})


def resolve_dataset(cli_data: str | None, train_args: dict) -> Path:
    """Find data.yaml: --data, then the repo-local copy, then the recorded path.

    Deliberately does NOT substitute another dataset when none of the three exist.
    Validating against the wrong split would produce a number that looks like an
    answer and is not one (Rule 6).
    """
    candidates: list[Path] = []
    if cli_data:
        candidates.append(Path(cli_data))
    candidates.append(DEFAULT_DATASET / "data.yaml")
    recorded = train_args.get("data")
    if recorded:
        candidates.append(Path(recorded))

    for path in candidates:
        if path.exists():
            return path.resolve()

    print("[ERROR] The training-distribution dataset was not found. Tried:")
    for path in candidates:
        print(f"          {path}")
    print()
    print("        This split is gitignored and absent on a fresh clone. It is built by")
    print("        scripts/prepare_visible_road_public_dataset.py from Cityscapes + ACDC +")
    print("        IDD + Mapillary, all of which need registered downloads. Point --data at")
    print("        an existing copy rather than rebuilding it.")
    sys.exit(1)


def print_train_args(train_args: dict) -> None:
    keys = ("task", "model", "data", "epochs", "batch", "imgsz", "rect", "optimizer",
            "lr0", "overlap_mask", "mask_ratio", "single_cls", "fraction")
    print("Recorded training arguments (from the checkpoint itself)")
    print("-" * 78)
    for key in keys:
        if key in train_args:
            print(f"  {key:<14} {train_args[key]}")
    print()


def run_val(model: YOLO, data_yaml: Path, imgsz: int, batch: int, device: str | None) -> object:
    """Val-mode metrics. Returns the ultralytics metrics object."""
    print("=" * 78)
    print("[1] VAL-MODE METRICS, PER CLASS")
    print("=" * 78)
    return model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        verbose=False,
    )


def report_per_class(metrics, names: dict) -> dict:
    """Print the per-class table and return the road class's row, if it was scored."""
    ap_index = list(getattr(metrics, "ap_class_index", []))
    if not ap_index:
        print("  No classes were scored -- the val split produced no labels.")
        return {}

    header = f"  {'class':<18} {'P(B)':>7} {'R(B)':>7} {'mAP50(B)':>9} {'mAP50-95(B)':>12}"
    has_seg = getattr(metrics, "seg", None) is not None
    if has_seg:
        header += f" {'P(M)':>7} {'R(M)':>7} {'mAP50(M)':>9} {'mAP50-95(M)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    road_row: dict = {}
    for i, class_id in enumerate(ap_index):
        name = names.get(int(class_id), str(class_id))
        bp, br, bap50, bap = metrics.box.class_result(i)
        line = f"  {name:<18} {bp:7.4f} {br:7.4f} {bap50:9.4f} {bap:12.4f}"
        row = {"precision_box": bp, "recall_box": br, "mAP50_box": bap50, "mAP50_95_box": bap}
        if has_seg:
            mp, mr, map50, mapx = metrics.seg.class_result(i)
            line += f" {mp:7.4f} {mr:7.4f} {map50:9.4f} {mapx:12.4f}"
            row.update({"precision_mask": mp, "recall_mask": mr,
                        "mAP50_mask": map50, "mAP50_95_mask": mapx})
        print(line)
        if name == ROAD_CLASS_NAME:
            road_row = row

    print()
    if not road_row:
        print(f"  !! {ROAD_CLASS_NAME} was NOT scored -- it has no labelled instances in this")
        print("     val split. That alone would explain issue #30.")
    return road_row


def sweep_predict(model: YOLO, data_yaml: Path, imgsz: int, max_images: int,
                  device: str | None, road_class_id: int) -> dict:
    """Predict-mode sweep. Counts images where the road class actually survives NMS."""
    print("=" * 78)
    print("[2] PREDICT-MODE SWEEP OVER THE SAME VAL IMAGES")
    print("=" * 78)

    images = sorted((data_yaml.parent / "images" / "val").glob("*.jpg"))[:max_images]
    if not images:
        print(f"  No val images under {data_yaml.parent / 'images' / 'val'}")
        return {}

    print(f"  {len(images)} images, road class id {road_class_id} ({ROAD_CLASS_NAME})\n")
    print(f"  {'conf':>7} {'imgs with road':>16} {'road instances':>16} {'max road conf':>15}")
    print("  " + "-" * 58)

    out = {}
    for conf in PREDICT_CONFS:
        hits = 0
        instances = 0
        best = 0.0
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            result = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            road = confs[cls_ids == road_class_id]
            if road.size:
                hits += 1
                instances += int(road.size)
                best = max(best, float(road.max()))
        print(f"  {conf:7.3f} {hits:>10}/{len(images):<5} {instances:>16} {best:>15.4f}")
        out[str(conf)] = {"images_with_road": hits, "images": len(images),
                          "road_instances": instances, "max_road_conf": best}

    print()
    print("  The 0.25 row is what the app sees: get_road_mask() calls the model with no")
    print("  conf argument, so ultralytics' predict default applies. A healthy 0.001 row")
    print("  with an empty 0.25 row means the gate, not the checkpoint, is the problem.")
    print()
    return out


def check_fallback_rate(data_yaml: Path, max_images: int) -> dict:
    """How often the lower-60% crop fires in-distribution -- issue #30's symptom."""
    print("=" * 78)
    print("[3] get_road_mask() FALLBACK RATE, IN-DISTRIBUTION")
    print("=" * 78)

    sys.path.insert(0, str(ROOT))
    from app.two_stage_detection import TwoStageDetector

    detector = TwoStageDetector(MODEL_DIR / "best.pt", MODEL_DIR / "road_seg.pt")
    if not detector.use_road_seg:
        print("  Road segmentation is not enabled -- road_seg.pt failed to load.")
        return {}

    images = sorted((data_yaml.parent / "images" / "val").glob("*.jpg"))[:max_images]
    fallbacks = 0
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        # A real mask and the fallback are both non-empty, so the mask alone cannot
        # tell them apart. Re-derive the road mask the same way get_road_mask() does
        # and check whether anything survived before the crop was substituted.
        result = detector.road_model(frame, verbose=False)[0]
        real_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if result.masks is not None and len(result.masks) > 0 and result.boxes is not None:
            include_ids = detector._resolve_class_ids(result.names, detector.road_include_keywords)
            exclude_ids = detector._resolve_class_ids(result.names, detector.road_exclude_keywords)
            keep = include_ids - exclude_ids
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(result.masks)):
                if i < len(cls_ids) and cls_ids[i] in keep:
                    h, w = frame.shape[:2]
                    real_mask = cv2.bitwise_or(real_mask, detector._extract_mask(result.masks, i, w, h))
        if real_mask.max() == 0:
            fallbacks += 1

    total = len(images)
    print(f"  lower-60% crop substituted: {fallbacks}/{total}")
    if total and fallbacks == total:
        print("  Every frame fell through -- the fault is NOT dash-camera geometry.")
    elif fallbacks == 0:
        print("  Never fell through -- the checkpoint works in-distribution; issue #30 is a")
        print("  domain gap, and the geometric prior is the honest Stage 1 for dash-cam input.")
    print()
    return {"fallbacks": fallbacks, "images": total}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate road_seg.pt on its own training-distribution val split")
    parser.add_argument("--model", default=str(MODEL_DIR / "road_seg.pt"))
    parser.add_argument("--data", default=None,
                        help="data.yaml of the visible-road dataset (default: repo copy, "
                             "then the path recorded in the checkpoint)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="default: the imgsz recorded in the checkpoint")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    parser.add_argument("--max-predict", type=int, default=100,
                        help="images for the predict sweep and fallback check")
    parser.add_argument("--json-out", default=None, help="write the numbers to this path")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    train_args = read_checkpoint_args(model_path)
    print_train_args(train_args)

    data_yaml = resolve_dataset(args.data, train_args)
    imgsz = args.imgsz or int(train_args.get("imgsz", 640))
    print(f"Dataset: {data_yaml}")
    print(f"imgsz:   {imgsz}{'  (from checkpoint)' if args.imgsz is None else '  (overridden)'}")
    print()

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    road_class_id = next((i for i, n in names.items() if n == ROAD_CLASS_NAME), None)
    if road_class_id is None:
        print(f"[ERROR] {ROAD_CLASS_NAME} is not among this model's classes: {names}")
        sys.exit(1)

    metrics = run_val(model, data_yaml, imgsz, args.batch, args.device)
    road_row = report_per_class(metrics, names)
    predict_rows = sweep_predict(model, data_yaml, imgsz, args.max_predict,
                                 args.device, road_class_id)
    fallback = check_fallback_rate(data_yaml, args.max_predict)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if road_row and road_row.get("mAP50_box", 0.0) > 0.2:
        print(f"  {ROAD_CLASS_NAME} scores mAP50(B)={road_row['mAP50_box']:.4f} in-distribution.")
        print("  The checkpoint is not broken. Compare rows [2] and [3] to decide whether the")
        print("  dash-cam failure is a domain gap or the predict-mode confidence gate.")
    elif road_row:
        print(f"  {ROAD_CLASS_NAME} scores mAP50(B)={road_row['mAP50_box']:.4f} even in-distribution.")
        print("  The class is under-trained. Issue #30 is a training fault, not a domain gap,")
        print("  and no inference-side change will fix it.")
    else:
        print(f"  {ROAD_CLASS_NAME} was not scored at all. Check the val split's labels first.")
    print()

    if args.json_out:
        payload = {
            "model": str(model_path),
            "data": str(data_yaml),
            "imgsz": imgsz,
            "train_args": {k: str(v) for k, v in train_args.items()},
            "per_class_road": road_row,
            "predict_sweep": predict_rows,
            "fallback": fallback,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
