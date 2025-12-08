from __future__ import annotations

from typing import Dict, Any
import pandas as pd


def load_berkeley_dataset(
    csv_path: str,
    moteid: int,
    sample_period_seconds: int,
) -> pd.DataFrame:
    """
    Load Intel Berkeley Lab dataset from CSV with columns:
    date,time,epoch,moteid,temperature,humidity,light,voltage

    Returns DataFrame indexed by timestamp with:
      temperature_c, humidity_pct, light_lux
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=[
            "date",
            "time",
            "epoch",
            "moteid",
            "temperature",
            "humidity",
            "light",
            "voltage",
        ],
        delim_whitespace=False,
    )

    # Filter by moteid (one sensor)
    df = df[df["moteid"] == moteid].copy()

    # Build timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.set_index("timestamp").sort_index()

    out = pd.DataFrame(
        {
            "temperature_c": df["temperature"],
            "humidity_pct": df["humidity"],
            "light_lux": df["light"],
        }
    )

    # Resample to uniform step
    rule = f"{sample_period_seconds}s"
    out_res = out.resample(rule).mean().interpolate()

    return out_res
