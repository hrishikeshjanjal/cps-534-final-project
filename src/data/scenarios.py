from __future__ import annotations

from typing import Literal
import pandas as pd
import numpy as np


def _build_time_index(
    duration_seconds: int,
    sample_period_seconds: int,
    start_ts: str = "2020-01-01 18:00:00",
) -> pd.DatetimeIndex:
    steps = duration_seconds // sample_period_seconds
    return pd.date_range(
        start=pd.Timestamp(start_ts),
        periods=steps,
        freq=f"{sample_period_seconds}s",
    )


def generate_scenario_a(
    duration_seconds: int = 600,
    sample_period_seconds: int = 10,
) -> pd.DataFrame:
    """
    Scenario A: Evening dim light + user coming and going.
    Goal: show light turning on/off based on presence.
    """
    idx = _build_time_index(duration_seconds, sample_period_seconds)

    steps = len(idx)
    # Presence: present for 2 minutes, away for 1 minute repeatedly
    pattern = np.tile([1, 1, 1, 1, 0, 0], steps // 6 + 1)[:steps]

    df = pd.DataFrame(index=idx)
    df["present"] = pattern
    df["distance_cm"] = np.where(df["present"] == 1, 70.0, 200.0)
    df["light_lux"] = 60.0  # dim office
    df["temperature_c"] = 24.0
    df["humidity_pct"] = 45.0
    df["posture_score"] = 0.85  # generally good posture

    return df


def generate_scenario_b(
    duration_seconds: int = 600,
    sample_period_seconds: int = 10,
) -> pd.DataFrame:
    """
    Scenario B: User is present in a hot & humid environment.
    Goal: drive fan logic once temp/humidity thresholds are tuned.
    """
    idx = _build_time_index(duration_seconds, sample_period_seconds)
    steps = len(idx)

    df = pd.DataFrame(index=idx)
    df["present"] = 1
    df["distance_cm"] = 70.0

    # Temperature ramps from 25 to 30 C
    df["temperature_c"] = np.linspace(25.0, 30.0, steps)
    # Humidity around 70%
    df["humidity_pct"] = 70.0
    # Light comfortable
    df["light_lux"] = 350.0
    df["posture_score"] = 0.85

    return df


def generate_scenario_c(
    duration_seconds: int = 600,
    sample_period_seconds: int = 10,
) -> pd.DataFrame:
    """
    Scenario C: User sits for a while with worsening posture.
    Goal: trigger posture alerts when posture_score stays low.
    """
    idx = _build_time_index(duration_seconds, sample_period_seconds)
    steps = len(idx)

    df = pd.DataFrame(index=idx)
    df["present"] = 1
    df["distance_cm"] = 70.0
    df["light_lux"] = 300.0
    df["temperature_c"] = 24.0
    df["humidity_pct"] = 45.0

    # Posture score decays from 0.9 to 0.4
    df["posture_score"] = np.linspace(0.9, 0.4, steps)

    return df


def save_scenario_to_csv(df: pd.DataFrame, path: str) -> None:
    df_out = df.reset_index().rename(columns={"index": "timestamp"})
    df_out.to_csv(path, index=False)
