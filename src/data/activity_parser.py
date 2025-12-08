from __future__ import annotations

from typing import Set
import pandas as pd

# Treat these activities as "active"
ACTIVE_LABELS: Set[str] = {
    "Cooking",
    "Eat",
    "Eating",
    "Cleaning",
    "Work",
    "Working",
    "WorkingAtDesk",
    "Watching_TV",
    "WatchingTV",
    "Using_Computer",
    "UsingComputer",
    "Personal_Hygiene",
    "Bathing",
    "Showering",
    "Laundry",
    "Dishwashing",
    "Grooming",
    "Dressing",
    "Moving",
    "Walking",
    "Active",
    "Bed_Toilet_Transition",   # from your sample – clearly active
}

INACTIVE_LABELS: Set[str] = {
    "Sleeping",
    "Idle",
    "No_Activity",
    "NoActivity",
    "In_Bed",
    "InBed",
    "Away",
}


def extract_activity_from_ann_features(
    ann_features_path: str,
    output_csv_path: str,
    sample_period_seconds: int = 10,
) -> None:
    """
    Reads csh103.ann.features.csv and produces a smaller CSV with:
      timestamp (synthetic, monotonic),
      activity_label,
      is_active (0/1)
    """
    print(f"Loading annotated feature file: {ann_features_path}")
    df = pd.read_csv(ann_features_path)

    if "activity" not in df.columns:
        raise ValueError("Expected an 'activity' column in ann.features.csv")

    # Use the 'activity' column as label
    df["activity_label"] = df["activity"].astype(str)

    # Build synthetic timestamps: start at arbitrary date and step by sample_period_seconds
    base_ts = pd.Timestamp("2020-01-01 00:00:00")
    n = len(df)
    df["timestamp"] = base_ts + pd.to_timedelta(
        range(n), unit="s"
    ) * sample_period_seconds

    def to_active(label: str) -> int:
        if label in ACTIVE_LABELS:
            return 1
        if label in INACTIVE_LABELS:
            return 0
        # default: treat unknown as inactive
        return 0

    df["is_active"] = df["activity_label"].apply(to_active)

    out = df[["timestamp", "activity_label", "is_active"]].copy()
    out.to_csv(output_csv_path, index=False)

    print(f"Saved processed activity file → {output_csv_path}")
    print(out.head())
