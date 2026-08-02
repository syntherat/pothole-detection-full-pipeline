"""
integration/sensor_adapter.py

Wraps your existing PotholeDetector (physics FSM) and the sklearn AI model so the orchestrator can call one function per row and get back a score, not just a hard 0/1 label.

Import path assumption: this file sits in `integration/`, next to your
EPICS/ folder. Adjust sys.path if your layout differs.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
EPICS_DETECTOR_DIR = BASE_DIR / "pothole-detect-physics" / "detector_py"
sys.path.insert(0, str(EPICS_DETECTOR_DIR))

from pothole_detection import PotholeDetector  # noqa: E402

AI_MODEL_PATH = BASE_DIR / "pothole-detect-physics" / "Data" / "pothole_ai_model.pkl"

FEATURE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "speed"]


class SensorSession:
    """
    One instance per continuous drive/session.

    IMPORTANT: PotholeDetector is stateful (DROP -> FREEFALL -> IMPACT).
    Never share one instance across two different sessions/vehicles --
    create a fresh SensorSession() per session instead.
    """

    def __init__(self, ai_model_path: Path = AI_MODEL_PATH):
        self.physics_detector = PotholeDetector()
        self.ai_model = joblib.load(ai_model_path)

    def check_row(self, row: pd.Series) -> dict:
        """
        Run both physics + AI on a single sensor row.
        Returns a plain dict -- orchestrator maps this onto PotholeCandidateEvent.
        """
        physics_result = self.physics_detector.process_sample(
            timestamp=row["timestamp"],
            ax=row["ax"],
            ay=row["ay"],
            az=row["az"],
            gx=row["gx"],
            gy=row["gy"],
            gz=row["gz"],
            speed=row["speed"],
        )

        # predict_proba instead of predict -- gives a real score, not just 0/1
        features = pd.DataFrame([row[FEATURE_COLUMNS]])
        ai_score = float(self.ai_model.predict_proba(features)[0][1])

        return {
            "physics_detected": physics_result["pothole_detected"],
            "physics_depth_estimate": physics_result["depth_estimate"],
            "physics_length_estimate": physics_result["length_estimate"],
            "ai_score": ai_score,
        }