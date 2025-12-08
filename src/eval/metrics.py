from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def load_log(path: str = "logs/simulation_log.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    total_steps = len(df)
    metrics["total_steps"] = int(total_steps)

    # Latency
    if "latency_ms" in df.columns:
        metrics["avg_latency_ms"] = float(df["latency_ms"].mean())
        metrics["p95_latency_ms"] = float(df["latency_ms"].quantile(0.95))
    else:
        metrics["avg_latency_ms"] = None
        metrics["p95_latency_ms"] = None

    # Actions / decisions
    decisions = df["actions"].fillna("")
    metrics["num_decisions"] = int((decisions != "").sum())

    # Duty cycles
    metrics["light_on_ratio"] = float(df["light_on"].mean())
    metrics["fan_on_ratio"] = float(df["fan_on"].mean())
    metrics["posture_alert_ratio"] = float(df["posture_alert_on"].mean())

    # Energy
    metrics["total_energy_wh"] = float(df["energy_used_wh"].max())

    # Simple "uptime": fraction of rows without NaNs
    metrics["uptime_ratio"] = float(
        df.notna().all(axis=1).sum() / total_steps if total_steps > 0 else 0.0
    )

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n=== Simulation Metrics ===")
    for k, v in metrics.items():
        print(f"{k:22s}: {v}")
