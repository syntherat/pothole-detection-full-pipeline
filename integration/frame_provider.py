"""
	Fakes the missing frame↔sensor link and GPS until real synced data exists
"""

import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Point this at whatever folder currently holds sample road images
# (e.g. your YOLO val/test images, or any folder of road photos).
MOCK_IMAGE_DIR = BASE_DIR / "pothole_detection_app" / "data" / "sample_images"

# Simulated route start point (swap for your test location)
START_LAT, START_LNG = 23.2599, 77.4126
METERS_PER_SECOND_LAT = 0.000009   # rough conversion, fine for a demo


def _mock_image_pool():
    if not MOCK_IMAGE_DIR.exists():
        return []
    return list(MOCK_IMAGE_DIR.glob("*.jpg")) + list(MOCK_IMAGE_DIR.glob("*.png"))


def get_mock_frame(event_id: str) -> str:
    """
    Returns a path to a stand-in image. Deterministic per event_id so
    repeated runs are reproducible, not random each time.
    """
    pool = _mock_image_pool()
    if not pool:
        raise FileNotFoundError(
            f"No sample images found in {MOCK_IMAGE_DIR}. "
            "Add a handful of road/pothole photos there for testing, "
            "or point MOCK_IMAGE_DIR at an existing dataset folder."
        )
    index = hash(event_id) % len(pool)
    return str(pool[index])


def simulate_gps(timestamp: float) -> tuple[float, float]:
    """
    Fake but consistent GPS point based on elapsed time.
    Replace with real GPS reads once available.
    """
    lat = START_LAT + timestamp * METERS_PER_SECOND_LAT
    lng = START_LNG
    return lat, lng


class MockFrameProvider:
    """
    Object wrapper around the two functions above, so the orchestrator can take
    a provider as a parameter and the mock and the real CARLA provider are
    interchangeable.

    The functions themselves are left untouched -- test_integration.py calls
    them directly.

    Note the asymmetry, which is the whole reason this is a mock: get_frame()
    ignores the timestamp and keys on event_id, because there is no real
    relationship between a sensor row and any image in the pool. The CARLA
    provider does the opposite, and keying on time is what makes it real.
    See integration/carla_frame_provider.py.
    """

    def get_frame(self, timestamp: float, event_id: str | None = None) -> str:
        return get_mock_frame(event_id if event_id is not None else str(timestamp))

    def get_gps(self, timestamp: float) -> tuple[float, float]:
        return simulate_gps(timestamp)