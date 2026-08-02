"""
integration/vision_adapter.py

Wraps TwoStageDetector so the orchestrator can pass in a frame (numpy array
or image path) and get back a simple confidence score, instead of a raw
YOLO Results object.

Uses TwoStageDetector.detect_potholes() directly (NOT utils.run_detection),
because run_detection() also saves/annotates images to disk -- more than
we want mid-pipeline. We only annotate/save once a detection is confirmed.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
VISION_APP_DIR = BASE_DIR / "pothole_detect_app" / "app"
sys.path.insert(0, str(BASE_DIR))

from pothole_detect_app.app.two_stage_detection import create_two_stage_detector  # noqa: E402

POTHOLE_MODEL_PATH = BASE_DIR / "pothole_detect_app" / "model" / "best.pt"
ROAD_MODEL_PATH = BASE_DIR / "pothole_detect_app" / "model" / "road_seg.pt"


class VisionSession:
    """One instance can be reused across the whole run -- YOLO models are stateless."""

    def __init__(self):
        self.detector = create_two_stage_detector(
            str(POTHOLE_MODEL_PATH),
            str(ROAD_MODEL_PATH) if ROAD_MODEL_PATH.exists() else None,
        )

    def confirm(self, frame_or_path, conf: float = 0.35) -> dict:
        """
        frame_or_path: either a numpy BGR frame, or a str/Path to an image file.
        Returns dict with vision_score and vision_confirmed.
        """
        if isinstance(frame_or_path, (str, Path)):
            frame = cv2.imread(str(frame_or_path))
            if frame is None:
                raise ValueError(f"Failed to read image: {frame_or_path}")
        else:
            frame = frame_or_path

        results = self.detector.detect_potholes(frame, conf=conf)

        if results.boxes is not None and len(results.boxes) > 0:
            confidences = results.boxes.conf.cpu().numpy().tolist()
            vision_score = float(np.mean(confidences))
        else:
            vision_score = 0.0

        return {
            "vision_score": vision_score,
            "vision_confirmed": vision_score >= conf,
        }