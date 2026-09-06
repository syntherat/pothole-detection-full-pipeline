"""
Trains the RandomForest that filters the physics detector's candidates.

Revised 2026-09-06 (stages A and B of PAVE_v2_change_log_REVIEWED.docx):

  * the train/test split is grouped by event, not by row (item 4.1)
  * physics-aware rolling features are available (item 4.3)
  * both the leaky and the honest score are reported side by side

The model stays a RandomForest. Swapping to LightGBM was assessed and rejected:
it adds a compiled dependency to a deliberately light requirements file and
invalidates the scikit-learn pin that exists to match the pickled model, while
the measured problem was leakage rather than model capacity.

Usage:
    python Model/train_ai_model.py              # raw 7 features -> v1 model path
    python Model/train_ai_model.py --rolling    # + rolling features -> v2 path
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import features as feat


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Data" / "synthetic_pothole_dataset.csv"

# The raw-feature model keeps the original filename, because
# integration/sensor_adapter.py loads exactly this path and feeds it exactly the
# seven raw columns.
MODEL_PATH = BASE_DIR / "Data" / "pothole_ai_model.pkl"

# The rolling-feature model gets its own file. It CANNOT be dropped in as a
# replacement: the adapter builds a one-row DataFrame per sample and keeps no
# history, so a model expecting rolling windows would be handed columns that do
# not exist. Wiring it in is a separate change to the integration layer.
MODEL_PATH_ROLLING = BASE_DIR / "Data" / "pothole_ai_model_rolling.pkl"


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling", action="store_true",
                        help="Add the physics-aware rolling features (item 4.3).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override the output model path.")
    args = parser.parse_args()

    data = pd.read_csv(DATASET_PATH)
    data = feat.ensure_event_ids(data)

    if args.rolling:
        data = feat.add_rolling_features(data)
        feature_columns = feat.full_feature_columns()
        default_out = MODEL_PATH_ROLLING
    else:
        feature_columns = list(feat.RAW_FEATURE_COLUMNS)
        default_out = MODEL_PATH

    out_path = args.out or default_out

    X = data[feature_columns]
    y = data["label"]
    groups = feat.grouping_key(data)

    n_events = data.loc[data["event_id"] >= 0, "event_id"].nunique()
    print(f"Rows: {len(data):,} | labelled events: {n_events} | features: {len(feature_columns)}")

    # ---------------------------------------------------------------
    # The old split, reported for comparison only -- never used to train
    # ---------------------------------------------------------------
    # Each pothole spans twelve consecutive rows. Splitting by row puts some of
    # those rows in train and the rest in test, so the model is evaluated on the
    # remainder of events it has already partly seen. The number this produces
    # is not a generalisation estimate. It is printed because the honest figure
    # is only interpretable next to the inflated one it replaces.
    Xtr_leaky, Xte_leaky, ytr_leaky, yte_leaky = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    leaky = build_model()
    leaky.fit(Xtr_leaky, ytr_leaky)
    leaky_pred = leaky.predict(Xte_leaky)
    leaky_acc = accuracy_score(yte_leaky, leaky_pred)
    # Accuracy is close to useless here -- roughly 97 % of rows are negative, so
    # a model that never predicts a pothole still scores ~97 %. Positive-class
    # recall is where leakage actually shows itself, so it is reported too.
    leaky_recall = recall_score(yte_leaky, leaky_pred, pos_label=1, zero_division=0)
    leaky_prec = precision_score(yte_leaky, leaky_pred, pos_label=1, zero_division=0)

    # ---------------------------------------------------------------
    # The honest split: no event appears on both sides
    # ---------------------------------------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Cheap assertion, because a silently broken split is the exact failure this
    # change exists to prevent.
    overlap = {g for g in groups[train_idx] if g >= 0} & {g for g in groups[test_idx] if g >= 0}
    assert not overlap, f"event(s) {sorted(overlap)[:5]} span both splits -- grouping is broken"

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    honest_acc = accuracy_score(y_test, predictions)
    honest_recall = recall_score(y_test, predictions, pos_label=1, zero_division=0)
    honest_prec = precision_score(y_test, predictions, pos_label=1, zero_division=0)

    joblib.dump(model, out_path)

    print("\nAI Model Performance (grouped split, honest):\n")
    print(classification_report(y_test, predictions, zero_division=0))

    print("=" * 70)
    print(f"  {'metric':<26} {'leaky (row split)':>18} {'honest (grouped)':>18}")
    print("-" * 70)
    print(f"  {'accuracy':<26} {leaky_acc * 100:>17.2f}% {honest_acc * 100:>17.2f}%")
    print(f"  {'recall (pothole class)':<26} {leaky_recall * 100:>17.2f}% {honest_recall * 100:>17.2f}%")
    print(f"  {'precision (pothole class)':<26} {leaky_prec * 100:>17.2f}% {honest_prec * 100:>17.2f}%")
    print("=" * 70)
    print("Read the recall row, not the accuracy row. About 97 % of rows are")
    print("negative, so accuracy stays near 99 % whatever the model does with")
    print("potholes -- it is the metric least able to show the leak.")
    print("A fall is the expected result, not a regression: the leaky column was")
    print("measured on events the model had already partly trained on.")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
