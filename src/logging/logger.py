from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from src.core.state import DeskState


class SimulationLogger:
    """
    Logs each simulation step to a CSV file.
    """

    def __init__(self, log_path: str = "logs/simulation_log.csv") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = self.log_path.open("w", newline="")
        fieldnames = [
            "timestamp",
            "present",
            "distance_cm",
            "light_lux",
            "temperature_c",
            "humidity_pct",
            "posture_score",
            "light_on",
            "fan_on",
            "posture_alert_on",
            "energy_used_wh",
            "actions",
            "latency_ms",
            "llm_message",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log_step(
        self,
        timestamp,
        sensor: Dict,
        state: DeskState,
        actions: List[str],
        latency_ms: float,
        llm_message: str,
    ) -> None:
        row = {
            "timestamp": timestamp.isoformat(),
            "present": int(bool(sensor.get("present", False))),
            "distance_cm": float(sensor.get("distance_cm", 0.0)),
            "light_lux": float(sensor.get("light_lux", 0.0)),
            "temperature_c": float(sensor.get("temperature_c", 0.0)),
            "humidity_pct": float(sensor.get("humidity_pct", 0.0)),
            "posture_score": float(sensor.get("posture_score", 1.0)),
            "light_on": int(state.light_on),
            "fan_on": int(state.fan_on),
            "posture_alert_on": int(state.posture_alert_on),
            "energy_used_wh": float(state.energy_used_wh),
            "actions": ";".join(actions),
            "latency_ms": float(latency_ms),
            "llm_message": llm_message,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
