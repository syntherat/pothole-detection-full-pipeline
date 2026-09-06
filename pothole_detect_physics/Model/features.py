"""
Shared feature construction for training and evaluation.

WHY THIS MODULE EXISTS
----------------------
train_ai_model.py and run_detector_on_dataset.py must build **identical**
features. When each script built its own, any divergence would show up as a
mysterious accuracy drop at evaluation time rather than as an error, so the
construction lives in one place and both import it.

Added 2026-09-06 for stages A and B of PAVE_v2_change_log_REVIEWED.docx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The seven raw channels the sensor actually produces. This is also what
# integration/sensor_adapter.py feeds the live model, so any model trained on
# more than these cannot be dropped into the cascade without teaching the
# adapter to keep history. See ROLLING_FEATURE_COLUMNS below.
RAW_FEATURE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "speed"]

G = 9.81

# Window lengths in SAMPLES, interpreted at 400 Hz -> 20 ms, 50 ms, 100 ms.
# The source document specified 8/20/40 at 1 kHz; the repository runs at 400 Hz
# throughout, so the counts are kept and their durations differ. Do not port
# these to another sampling rate without rescaling -- an 8-sample window means
# something different at every rate.
ROLLING_WINDOWS = (8, 20, 40)


def ensure_event_ids(data: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee `event_id` and `event_type` columns.

    Datasets written by the current generate_dataset.py carry them. The dataset
    committed to the repository predates that and does not, and it cannot simply
    be regenerated -- generate_dataset.py was unseeded when that file was
    written, so a regeneration produces different data and invalidates the
    pickled model trained on it.

    For those older files the ids are derived from runs of consecutive
    `label == 1`, which is exactly how integration/test_integration.py's
    group_label_events() defines an event. Same definition, same unit.

    Background rows get a **unique negative id each**, not one shared id. A
    single shared group would force every negative row into either the train or
    the test side of a grouped split, which quietly destroys the split.
    """
    data = data.copy()

    if "event_id" in data.columns:
        if "event_type" not in data.columns:
            data["event_type"] = np.where(data["label"] == 1, "pothole", "none")
        return data

    labels = data["label"].to_numpy()
    event_id = np.full(len(labels), -1, dtype=np.int64)

    in_event = False
    next_id = 0
    for i, lab in enumerate(labels):
        if lab == 1 and not in_event:
            in_event = True
            start = i
        elif lab != 1 and in_event:
            in_event = False
            event_id[start:i] = next_id
            next_id += 1
    if in_event:
        event_id[start:] = next_id

    data["event_id"] = event_id
    data["event_type"] = np.where(event_id >= 0, "pothole", "none")
    return data


def grouping_key(data: pd.DataFrame) -> np.ndarray:
    """
    Group vector for GroupShuffleSplit.

    Labelled events keep their `event_id`; every background row becomes its own
    group (encoded as a distinct negative number). That keeps all twelve rows of
    a pothole on the same side of the split -- the actual leak -- without
    collapsing the negatives into one indivisible block.
    """
    ids = data["event_id"].to_numpy().copy()
    background = ids < 0
    ids[background] = -(np.arange(background.sum(), dtype=np.int64) + 1)
    return ids


def add_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-aware context features (stage B / item 4.3).

    The raw model classifies each row in isolation, but a pothole is a temporal
    pattern -- the defining information lies in the rows around it. These add
    that context without changing the model family.

    All rolling windows are **causal** (`center=False`): a detector running live
    cannot see the future, and a centred window would leak it into training and
    inflate the score in a way that never survives deployment.
    """
    data = data.copy()

    az = data["az"]
    data["az_minus_g"] = az - G
    data["az_abs"] = az.abs()
    data["az_diff_1"] = az.diff().fillna(0.0)
    data["az_diff_2"] = az.diff().diff().fillna(0.0)

    for w in ROLLING_WINDOWS:
        roll = az.rolling(window=w, min_periods=1)
        data[f"az_min_{w}"] = roll.min()
        data[f"az_max_{w}"] = roll.max()
        data[f"az_std_{w}"] = roll.std().fillna(0.0)

    data["gyro_norm"] = np.sqrt(
        data["gx"] ** 2 + data["gy"] ** 2 + data["gz"] ** 2
    )
    return data


def rolling_feature_columns() -> list[str]:
    cols = ["az_minus_g", "az_abs", "az_diff_1", "az_diff_2"]
    for w in ROLLING_WINDOWS:
        cols += [f"az_min_{w}", f"az_max_{w}", f"az_std_{w}"]
    cols.append("gyro_norm")
    return cols


def full_feature_columns() -> list[str]:
    return RAW_FEATURE_COLUMNS + rolling_feature_columns()
