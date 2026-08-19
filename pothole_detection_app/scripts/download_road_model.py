# scripts/download_road_model.py
"""
Download a generic YOLOv8 segmentation model as a LAST-RESORT stand-in for
model/road_seg.pt.

READ THIS BEFORE RUNNING.

This repo already ships a real road segmentation model: a 7-class yolo11s-seg
trained for visible-road masking (visible_road, vehicle, pedestrian, shadow,
vegetation, roadside_object, road_obstacle). This script downloads
`yolov8s-seg.pt`, which is trained on **COCO** -- and COCO has **no road class
at all**. TwoStageDetector will find no road-like class names, fall through to
its "lower two-thirds of the frame" heuristic, and mask far worse.

So this is a downgrade, not an upgrade. It exists only for the case where
road_seg.pt is missing entirely and you need *something* so the two-stage code
path runs. To build a real model, use:

    scripts/prepare_visible_road_public_dataset.py   (build the dataset)
    scripts/train_multiclass_road_seg.py             (train it)

Because overwriting a real model with a COCO one is silent and hard to notice,
this script REFUSES to overwrite an existing file unless you pass --force.

Usage:
    python scripts/download_road_model.py                  # aborts if road_seg.pt exists
    python scripts/download_road_model.py --output model/road_seg_coco.pt
    python scripts/download_road_model.py --force          # overwrite, deliberately
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "model" / "road_seg.pt"
SOURCE_WEIGHTS = "yolov8s-seg.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a generic COCO segmentation model as a stand-in for road_seg.pt."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Where to write the weights (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output file if it already exists. Think first: the shipped "
             "road_seg.pt is a real 7-class visible-road model and this one is not.",
    )
    return parser.parse_args()


def download_road_segmentation_model(output: Path = DEFAULT_OUTPUT, force: bool = False):
    """
    Fetch SOURCE_WEIGHTS and copy them to `output`.

    Returns the loaded model, or None if nothing was written.
    """
    # Check BEFORE downloading -- no point fetching weights we are going to refuse
    # to install, and no chance of a half-finished overwrite.
    if output.exists() and not force:
        size_mb = output.stat().st_size / (1024 * 1024)
        print("ABORTED: refusing to overwrite an existing model.")
        print()
        print(f"  {output}")
        print(f"  {size_mb:.1f} MB already on disk.")
        print()
        print("  The model shipped with this repo is a 7-class yolo11s-seg trained for")
        print("  visible-road masking. What this script downloads is COCO-trained and has")
        print("  NO road class, so installing it over the top would silently degrade every")
        print("  road mask to a crude 'lower two-thirds of the frame' heuristic.")
        print()
        print("  If you still want it, either:")
        print(f"    --output {output.parent / 'road_seg_coco.pt'}   (keep both)")
        print("    --force                                          (overwrite anyway)")
        return None

    print(f"Downloading {SOURCE_WEIGHTS} (COCO-trained, general purpose)...")

    # Imported here so --help and the abort path above work without ultralytics
    # installed, and without triggering a model download on import.
    from ultralytics import YOLO

    model = YOLO(SOURCE_WEIGHTS)

    output.parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics drops the file either next to the caller or in its cache.
    candidates = [
        ROOT / SOURCE_WEIGHTS,
        Path.cwd() / SOURCE_WEIGHTS,
        Path.home() / ".cache" / "ultralytics" / SOURCE_WEIGHTS,
    ]
    source = next((c for c in candidates if c.exists()), None)

    if source is None:
        print(f"FAILED: could not find {SOURCE_WEIGHTS} after download.")
        print("  Looked in:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Copy it to {output} by hand if you have it.")
        return None

    if output.exists():
        backup = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup)
        print(f"  Existing model backed up to: {backup}")

    shutil.copy2(source, output)
    print(f"OK: written to {output}")
    print(f"  source: {source}")

    print()
    print("REMINDER: this is a COCO model. It has no 'road' class, so")
    print("TwoStageDetector's keyword matching will find nothing and fall back to its")
    print("frame-position heuristic. Treat the masks it produces with suspicion.")

    return model


def main() -> int:
    args = _parse_args()
    result = download_road_segmentation_model(output=args.output, force=args.force)

    if result is None:
        return 1

    print()
    print("=" * 60)
    print("Next steps:")
    print("1. Test the model: python scripts/test_road_segmentation.py")
    print("2. For a real visible-road model, build a dataset with")
    print("   scripts/prepare_visible_road_public_dataset.py and train with")
    print("   scripts/train_multiclass_road_seg.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
