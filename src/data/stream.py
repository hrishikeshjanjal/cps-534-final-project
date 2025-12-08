from __future__ import annotations

from typing import Dict, Any, Iterator

import pandas as pd

from .berkeley_loader import load_berkeley_dataset


def load_activity_series(
    activity_csv_path: str,
    length: int,
) -> pd.Series:
    """
    Load processed activity CSV (timestamp, is_active) and return a
    0/1 Series of given length (trim or pad as needed).
    """
    df = pd.read_csv(activity_csv_path)
    s = df["is_active"].astype(int)

    if len(s) >= length:
        s = s.iloc[:length]
    else:
        # pad with zeros (inactive) to match length
        pad = [0] * (length - len(s))
        s = pd.concat([s, pd.Series(pad)], ignore_index=True)
    return s


def build_combined_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    sampling_cfg = config.get("sampling", {})
    sample_period = int(sampling_cfg.get("period_seconds", 10))
    duration = int(sampling_cfg.get("duration_seconds", 3600))

    ber_cfg = config.get("berkeley", {})
    act_cfg = config.get("activity", {})

    berkeley_path = ber_cfg.get("csv_path")
    moteid = int(ber_cfg.get("moteid", 1))
    activity_path = act_cfg.get("processed_activity_csv")

    # ENVIRONMENT DATA (Berkeley)
    env_df = load_berkeley_dataset(
        csv_path=berkeley_path,
        moteid=moteid,
        sample_period_seconds=sample_period,
    )

    max_steps = min(len(env_df), duration // sample_period)
    env_df = env_df.iloc[:max_steps].copy()

    # ACTIVITY SERIES (from csh103)
    act_series = load_activity_series(activity_path, length=max_steps)

    # Attach activity to env_df index
    env_df["is_active"] = act_series.values
    env_df["present"] = env_df["is_active"]

    return env_df


def get_sensor_stream(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Generator over merged Berkeley + activity data.
    """
    df = build_combined_dataframe(config)

    for ts, row in df.iterrows():
        yield {
            "timestamp": ts.to_pydatetime(),
            "present": bool(row.get("present", 0)),
            "distance_cm": 70.0,
            "light_lux": float(row.get("light_lux", 0.0)),
            "temperature_c": float(row.get("temperature_c", 0.0)),
            "humidity_pct": float(row.get("humidity_pct", 0.0)),
            "posture_score": 0.8,
        }
