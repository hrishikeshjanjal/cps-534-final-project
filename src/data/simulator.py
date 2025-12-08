from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def simulate_sensor_data(
    duration_seconds: int = 3600,
    sample_period_seconds: int = 10,
    base_light_day: float = 500.0,
    base_temp: float = 25.0,
    base_humidity: float = 55.0,
    noise_scale: float = 0.1,
) -> pd.DataFrame:
    """
    Generate synthetic time-series data for:
    present, distance_cm, light_lux, temperature_c, humidity_pct, posture_score
    """
    num_steps = duration_seconds // sample_period_seconds

    # Time index
    idx = pd.date_range(
        start=pd.Timestamp.utcnow(),
        periods=num_steps,
        freq=f"{sample_period_seconds}s",
    )

    np.random.seed(42)

    # Random presence pattern (80% of time present)
    present_pattern = np.random.rand(num_steps) > 0.2

    # Distance: close if present, far otherwise
    distance_cm = np.where(
        present_pattern,
        np.random.normal(60, 10, num_steps),
        np.random.normal(200, 20, num_steps),
    )

    # Light, temperature, humidity with noise
    light = base_light_day + np.random.normal(
        0, base_light_day * noise_scale, num_steps
    )
    temp = base_temp + np.random.normal(0, base_temp * noise_scale, num_steps)
    humid = base_humidity + np.random.normal(
        0, base_humidity * noise_scale, num_steps
    )

    # Posture score between 0 and 1 (1 = perfect posture)
    posture = np.clip(
        np.random.normal(0.8, 0.15, num_steps),
        0.0,
        1.0,
    )

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "present": present_pattern.astype(int),
            "distance_cm": distance_cm,
            "light_lux": light,
            "temperature_c": temp,
            "humidity_pct": humid,
            "posture_score": posture,
        }
    )

    return df.set_index("timestamp")
