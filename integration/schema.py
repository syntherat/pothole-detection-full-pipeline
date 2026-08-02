"""
integration/schema.py

Single shared object that both the sensor pipeline and the vision pipeline read/write. Nothing else in the codebase needs to agree on variable names --
every adapter converts its own internals into this shape.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PotholeCandidateEvent:
    # --- identity ---
    event_id: str
    timestamp: float                  # seconds, from sensor CSV clock

    # --- location (simulated for now, see frame_provider.py) ---
    lat: Optional[float] = None
    lng: Optional[float] = None

    # --- stage 1: sensor/physics + AI ---
    physics_detected: bool = False
    physics_depth_estimate: Optional[float] = None
    physics_length_estimate: Optional[float] = None
    ai_score: Optional[float] = None          # predict_proba, 0-1
    sensor_triggered: bool = False            # physics AND ai_score >= threshold

    # --- stage 2: vision ---
    frame_path: Optional[str] = None          # which image was used to confirm
    vision_score: Optional[float] = None      # avg YOLO confidence, 0-1
    vision_confirmed: Optional[bool] = None

    # --- fusion ---
    final_decision: Optional[bool] = None
    final_confidence: Optional[float] = None

    def as_dict(self):
        return self.__dict__.copy()