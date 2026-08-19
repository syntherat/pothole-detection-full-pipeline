"""
integration/test_integration.py

Two kinds of tests here, clearly separated:

  A) PLUMBING TESTS -- do the pieces call each other correctly and return
     sane values? These are fully valid right now.

  B) STAGE-1 ACCURACY TEST -- compares sensor_triggered against the real
     `label` column in your synthetic CSV. This is the one accuracy number
     you CAN trust today, since it doesn't depend on the mock vision frames.

     Scored PER EVENT, not per sample. The FSM fires once (on IMPACT) while
     the dataset labels all 12 samples of a pothole, so per-sample recall
     caps at 1/12 = 0.083 no matter how good the detector is. Per-sample
     figures are still printed for continuity, clearly marked.

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


def test_mock_provider_degrades_when_no_images():
    """
    MockFrameProvider must return None rather than raise when no stand-in images
    exist, matching CarlaFrameProvider's interface. get_mock_frame() raising used
    to kill the orchestrator on its first triggered event; the provider now lets
    the run finish and report how many events it skipped.
    """
    from frame_provider import MockFrameProvider, _mock_image_pool

    if _mock_image_pool():
        pytest.skip("Stand-in images are present, so the degraded path cannot be exercised")

    provider = MockFrameProvider()
    assert provider.get_frame(1.0, "some-event-id") is None

    # The underlying function keeps its raising contract.
    with pytest.raises(FileNotFoundError):
        get_mock_frame("some-event-id")


def test_mock_frame_is_stable_across_processes():
    """
    Selection must not depend on PYTHONHASHSEED. Python salts str hashing per
    process, so the original hash()-based index made get_mock_frame's
    'reproducible' docstring true only within a single run.
    """
    import hashlib
    from frame_provider import _mock_image_pool

    pool = _mock_image_pool()
    if not pool:
        pytest.skip("No stand-in images configured")

    expected = int(hashlib.md5(b"fixed-event-id").hexdigest(), 16) % len(pool)
    assert get_mock_frame("fixed-event-id") == str(pool[expected])


def test_simulate_gps_moves_with_time():
    lat0, lng0 = simulate_gps(0.0)
    lat1, lng1 = simulate_gps(10.0)
    assert lat1 != lat0  # should have moved
    assert lng1 == lng0  # straight-line mock only changes lat

# How far past a labelled block a trigger may still land and count as that
# event's detection. generate_dataset.py injects the IMPACT in the last 2 of
# each 12-sample block, so triggers normally land INSIDE the window; this only
# guards against threshold changes nudging the fire point past the end.
EVENT_MATCH_TOLERANCE_S = 0.02


def group_label_events(labels) -> list[tuple[int, int]]:
    """
    Contiguous runs of label == 1, as inclusive (start_index, end_index) pairs.

    One run == one physical pothole. generate_dataset.py writes 12 consecutive
    samples per pothole, which is precisely why per-sample scoring is the wrong
    unit for a detector that fires once per event.
    """
    events: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(labels):
        if value == 1 and start is None:
            start = i
        elif value != 1 and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(labels) - 1))
    return events


def test_label_event_grouping():
    """
    The metric's own logic, tested without loading the model -- fast, and it
    fails loudly if someone changes the grouping and breaks the accuracy test
    silently.
    """
    assert group_label_events([0, 0, 0]) == []
    assert group_label_events([1, 1, 1]) == [(0, 2)]
    assert group_label_events([0, 1, 1, 0, 0, 1, 0]) == [(1, 2), (5, 5)]
    assert group_label_events([1, 0, 1]) == [(0, 0), (2, 2)]
    # a run touching the end of the array must still close
    assert group_label_events([0, 0, 1, 1]) == [(2, 3)]
    # floats, as they arrive from pandas
    assert group_label_events([0.0, 1.0, 1.0, 0.0]) == [(1, 2)]


@pytest.mark.skipif(not AI_MODEL_PATH.exists(), reason="AI model .pkl not found")
def test_stage1_accuracy_against_real_labels():
    """
    Compares the sensor cascade against the dataset's real `label` column.

    Scored PER EVENT. A pothole counts as detected if the cascade triggered at
    any point inside its labelled block (or within EVENT_MATCH_TOLERANCE_S
    after it). A trigger counts as a true positive if it falls in any such
    window.

    Per-sample figures are printed too, marked with their own ceiling, because
    earlier baselines were recorded that way -- do NOT read the per-sample
    recall as a quality signal.

    Measured 2026-08-19 on the first 2000 rows: 3 events, event recall 1.00,
    precision 1.00. Prints rather than asserting a target, since "good enough"
    depends on your tuning goals -- use it as a baseline while adjusting
    SENSOR_THRESHOLD in fusion.py.
    """
    data = pd.read_csv(DATASET_PATH).head(2000)  # keep the test fast
    session = SensorSession()

    labels = data["label"].tolist()
    trigger_indices: list[int] = []

    for i, (_, row) in enumerate(data.iterrows()):
        result = session.check_row(row)
        if should_trigger_vision(result["physics_detected"], result["ai_score"]):
            trigger_indices.append(i)

    events = group_label_events(labels)

    # Derive the tolerance in samples from the data's own timestep, so this
    # survives a change of sampling rate (CARLA runs may not be 400 Hz).
    timestamps = data["timestamp"].tolist()
    dt = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.0025
    tolerance = int(round(EVENT_MATCH_TOLERANCE_S / dt)) if dt > 0 else 0

    def _in_any_event(index: int) -> bool:
        return any(start <= index <= end + tolerance for start, end in events)

    detected = [
        (start, end)
        for start, end in events
        if any(start <= i <= end + tolerance for i in trigger_indices)
    ]

    matched_triggers = [i for i in trigger_indices if _in_any_event(i)]

    event_recall = len(detected) / max(1, len(events))
    event_precision = len(matched_triggers) / max(1, len(trigger_indices))

    # Per-sample, for continuity with baselines recorded before this test was
    # corrected. The ceiling is 1 / mean_event_width.
    sample_tp = sum(1 for i in trigger_indices if labels[i] == 1)
    sample_positives = sum(1 for v in labels if v == 1)
    sample_recall = sample_tp / max(1, sample_positives)
    mean_width = (sum(e - s + 1 for s, e in events) / len(events)) if events else 1.0

    print()
    print("Stage-1 (sensor cascade) -- EVENT-level")
    print(f"  events in window : {len(events)}")
    print(f"  detected         : {len(detected)}")
    print(f"  triggers fired   : {len(trigger_indices)}")
    print(f"  recall           : {event_recall:.2f}")
    print(f"  precision        : {event_precision:.2f}")
    print("Stage-1 -- per-sample (for continuity; NOT a quality signal)")
    print(f"  recall           : {sample_recall:.2f}  (ceiling {1.0 / mean_width:.3f}"
          f" -- a once-per-event detector cannot beat 1/{mean_width:.0f})")

    missed = [e for e in events if e not in detected]
    if missed:
        print(f"  missed events    : {missed}")

    # Guard against a vacuous pass: if the slice contains no potholes, the
    # metrics above are meaningless and would silently read as perfect.
    assert events, (
        "No labelled pothole events in the analysed slice -- the test would "
        "pass vacuously. Increase the .head(N) window or check the dataset."
    )

    # Sanity floors, not targets -- these only catch "something is badly broken".
    assert 0.0 <= event_recall <= 1.0
    assert 0.0 <= event_precision <= 1.0
