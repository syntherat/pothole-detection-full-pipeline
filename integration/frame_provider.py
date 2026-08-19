"""
	Fakes the missing frame↔sensor link and GPS until real synced data exists
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
VISION_APP = BASE_DIR / "pothole_detection_app"

# The intended home for stand-in road photos. See its README for what to put here.
MOCK_IMAGE_DIR = VISION_APP / "data" / "sample_images"

# Searched in order; the first folder that actually contains images wins. The
# fallbacks mean anyone who has already run the training pipeline gets a working
# demo without configuring anything -- the empty sample_images folder was by far
# the most common reason the orchestrator failed on its first triggered event.
CANDIDATE_IMAGE_DIRS = [
    MOCK_IMAGE_DIR,
    VISION_APP / "input",
    VISION_APP / "data" / "dataset_v3" / "test" / "images",
    VISION_APP / "data" / "dataset_v3" / "val" / "images",
    VISION_APP / "data" / "dataset_v2" / "test" / "images",
    VISION_APP / "data" / "dataset_v2" / "val" / "images",
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Simulated route start point (swap for your test location)
START_LAT, START_LNG = 23.2599, 77.4126
METERS_PER_SECOND_LAT = 0.000009   # rough conversion, fine for a demo

_pool_source_logged = False


def _images_in(directory: Path) -> list[Path]:
    """Images directly inside `directory`, sorted for a stable index."""
    if not directory.is_dir():
        return []
    # Sorted because glob order is filesystem-dependent, and get_mock_frame()
    # indexes into this list -- unsorted means the same event_id could pick a
    # different image on a different machine.
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _mock_image_pool():
    """First candidate directory that contains images. Empty list if none do."""
    global _pool_source_logged
    for directory in CANDIDATE_IMAGE_DIRS:
        pool = _images_in(directory)
        if pool:
            if not _pool_source_logged:
                logger.info("Using %d stand-in image(s) from %s", len(pool), directory)
                _pool_source_logged = True
            return pool
    return []


def missing_images_message() -> str:
    """The advice to print when no stand-in images can be found anywhere."""
    searched = "\n".join(f"    {d}" for d in CANDIDATE_IMAGE_DIRS)
    return (
        f"No stand-in images found. Searched:\n{searched}\n"
        f"Add a handful of road/pothole photos to {MOCK_IMAGE_DIR} "
        f"(see the README there), or point MOCK_IMAGE_DIR at an existing folder."
    )


def get_mock_frame(event_id: str) -> str:
    """
    Returns a path to a stand-in image. Deterministic per event_id so
    repeated runs are reproducible, not random each time.

    Raises FileNotFoundError when no images exist anywhere. Callers that want to
    degrade rather than fail should use MockFrameProvider, which converts that
    into a None return.
    """
    pool = _mock_image_pool()
    if not pool:
        raise FileNotFoundError(missing_images_message())

    # md5 rather than hash(): Python salts str hashing per process, so hash()
    # made the "reproducible" promise above true only within a single run.
    digest = hashlib.md5(event_id.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(pool)
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

    _warned_missing = False

    def get_frame(self, timestamp: float, event_id: str | None = None) -> str | None:
        """
        None when no stand-in images exist, matching CarlaFrameProvider.

        get_mock_frame() raises in that case, which used to kill the orchestrator
        on its very first triggered event. Returning None instead lets the run
        finish and report how many events it had to skip.
        """
        try:
            return get_mock_frame(event_id if event_id is not None else str(timestamp))
        except FileNotFoundError as e:
            if not MockFrameProvider._warned_missing:
                logger.warning("%s", e)
                MockFrameProvider._warned_missing = True
            return None

    def get_gps(self, timestamp: float) -> tuple[float, float]:
        return simulate_gps(timestamp)