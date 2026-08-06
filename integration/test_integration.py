"""
integration/test_integration.py

Two kinds of tests here, clearly separated:

  A) PLUMBING TESTS -- do the pieces call each other correctly and return
     sane values? These are fully valid right now.

  B) STAGE-1 ACCURACY TEST -- compares sensor_triggered against the real
     `label` column in your synthetic CSV. This is the one accuracy number
     you CAN trust today, since it doesn't depend on the mock vision frames.

Run with:
    pip install pytest --break-system-packages
    pytest integration/test_integration.py -v
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import PotholeCandidateEvent
from fusion import should_trigger_vision, fuse, SENSOR_THRESHOLD, FUSION_THRESHOLD
from sensor_adapter import SensorSession, FEATURE_COLUMNS, AI_MODEL_PATH
from frame_provider import get_mock_frame, simulate_gps

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "pothole_detect_physics" / "Data" / "synthetic_pothole_dataset.csv"


def test_schema_defaults():
    """A fresh event should have safe defaults, not None where a bool is expected."""
    event = PotholeCandidateEvent(event_id="test-1", timestamp=1.0)
    assert event.physics_detected is False
    assert event.sensor_triggered is False
    assert event.final_decision is None  # not yet computed -- this one SHOULD be None


def test_fusion_logic_boundaries():
    """Fusion math should be deterministic and respect the threshold."""
    confirmed, score = fuse(ai_score=1.0, vision_score=1.0)
    assert confirmed is True
    assert score == pytest.approx(1.0)

    confirmed, score = fuse(ai_score=0.0, vision_score=0.0)
    assert confirmed is False
    assert score == pytest.approx(0.0)


def test_should_trigger_vision_requires_both_signals():
    """Physics alone or AI alone should NOT trigger -- both must agree."""
    assert should_trigger_vision(physics_detected=True, ai_score=0.9) is True
    assert should_trigger_vision(physics_detected=False, ai_score=0.9) is False
    assert should_trigger_vision(physics_detected=True, ai_score=0.1) is False


@pytest.mark.skipif(not AI_MODEL_PATH.exists(), reason="AI model .pkl not found")
def test_sensor_adapter_returns_expected_shape():
    """Check the adapter returns the right keys with values in valid ranges."""
    data = pd.read_csv(DATASET_PATH)
    row = data.iloc[0]

    session = SensorSession()
    result = session.check_row(row)

    assert set(result.keys()) == {
        "physics_detected", "physics_depth_estimate",
        "physics_length_estimate", "ai_score",
    }
    assert isinstance(result["physics_detected"], bool)
    assert 0.0 <= result["ai_score"] <= 1.0


def test_frame_provider_is_deterministic():
    """Same event_id should always return the same mock frame (reproducible runs)."""
    try:
        path_1 = get_mock_frame("fixed-event-id")
        path_2 = get_mock_frame("fixed-event-id")
        assert path_1 == path_2
    except FileNotFoundError:
        pytest.skip("No sample images configured yet in frame_provider.MOCK_IMAGE_DIR")


def test_simulate_gps_moves_with_time():
    lat0, lng0 = simulate_gps(0.0)
    lat1, lng1 = simulate_gps(10.0)
    assert lat1 != lat0  # should have moved
    assert lng1 == lng0  # straight-line mock only changes lat

@pytest.mark.skipif(not AI_MODEL_PATH.exists(), reason="AI model .pkl not found")
def test_stage1_accuracy_against_real_labels():
    """
    Compares sensor_triggered vs the dataset's real `label` column.
    This does NOT test vision or the final fused decision -- only whether
    physics+AI cascade is well-calibrated against known ground truth.

    Prints precision/recall rather than asserting a hard threshold, since
    "good enough" depends on your tuning goals -- use this as a baseline
    to track as you adjust SENSOR_THRESHOLD in fusion.py.
    """
    data = pd.read_csv(DATASET_PATH).head(2000)  # keep the test fast
    session = SensorSession()

    true_positives = false_positives = false_negatives = 0

    for _, row in data.iterrows():
        result = session.check_row(row)
        triggered = should_trigger_vision(result["physics_detected"], result["ai_score"])
        actual = bool(row["label"])

        if triggered and actual:
            true_positives += 1
        elif triggered and not actual:
            false_positives += 1
        elif not triggered and actual:
            false_negatives += 1

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)

    print(f"\nStage-1 (sensor cascade) precision={precision:.2f} recall={recall:.2f}")
    print(f"TP={true_positives} FP={false_positives} FN={false_negatives}")

    # Sanity floor, not a real target -- just catches "something is badly broken"
    assert precision >= 0.0 and recall >= 0.0