"""
	Fakes the missing frame↔sensor link and GPS until real synced data exists
"""

SENSOR_THRESHOLD = 0.5    # ai_score needed (alongside physics_detected) to trigger vision
VISION_THRESHOLD = 0.35   # matches your existing YOLO GUI default

SENSOR_WEIGHT = 0.4
VISION_WEIGHT = 0.6
FUSION_THRESHOLD = 0.5


def should_trigger_vision(physics_detected: bool, ai_score: float) -> bool:
    return physics_detected and ai_score >= SENSOR_THRESHOLD


def fuse(ai_score: float, vision_score: float) -> tuple[bool, float]:
    combined = SENSOR_WEIGHT * ai_score + VISION_WEIGHT * vision_score
    return combined >= FUSION_THRESHOLD, combined