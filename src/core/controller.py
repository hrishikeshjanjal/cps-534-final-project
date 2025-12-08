from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional

from .state import DeskState
from .rules import (
    decide_light_action,
    decide_fan_action,
    decide_posture_alert,
)


class Controller:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        energy_cfg = config.get("energy", {})
        self.energy_light_w = float(energy_cfg.get("light_w", 10.0))
        self.energy_fan_w = float(energy_cfg.get("fan_w", 30.0))

    def step(
        self,
        sensor: Dict[str, Any],
        state: DeskState,
        dt_seconds: float,
        now: Optional[datetime] = None,
    ) -> (DeskState, List[str]):
        if now is None:
            now = datetime.utcnow()

        actions: List[str] = []

        present = bool(sensor.get("present", False))
        light = float(sensor.get("light_lux", 0.0))
        temp = float(sensor.get("temperature_c", 0.0))
        humid = float(sensor.get("humidity_pct", 0.0))
        posture = float(sensor.get("posture_score", 1.0))

        thresholds = self.cfg.get("thresholds", {})
        light_thr = thresholds.get("light", {})
        temp_thr = thresholds.get("temperature", {})
        humid_thr = thresholds.get("humidity", {})
        posture_thr = thresholds.get("posture", {})

        la = decide_light_action(
            light_lux=light,
            present=present,
            on_threshold=light_thr.get("on_lux", 200.0),
            off_threshold=light_thr.get("off_lux", 400.0),
        )
        if la == "turn_on_light":
            if not state.light_on:
                state.light_on = True
                actions.append(la)
        elif la == "turn_off_light":
            if state.light_on:
                state.light_on = False
                actions.append(la)

        fa = decide_fan_action(
            temperature_c=temp,
            humidity_pct=humid,
            present=present,
            temp_on=temp_thr.get("on_c", 27.0),
            temp_off=temp_thr.get("off_c", 24.0),
            humid_on=humid_thr.get("on_pct", 65.0),
        )
        if fa == "turn_on_fan":
            if not state.fan_on:
                state.fan_on = True
                actions.append(fa)
        elif fa == "turn_off_fan":
            if state.fan_on:
                state.fan_on = False
                actions.append(fa)

        pa = decide_posture_alert(
            posture_score=posture,
            present=present,
            bad_posture_threshold=posture_thr.get("bad_threshold", 0.7),
        )
        if pa == "posture_alert_on":
            if not state.posture_alert_on:
                state.posture_alert_on = True
                actions.append(pa)
        elif pa == "clear_posture_alert":
            if state.posture_alert_on:
                state.posture_alert_on = False
                actions.append(pa)

        # Energy
        hours = dt_seconds / 3600.0
        if state.light_on:
            state.energy_used_wh += self.energy_light_w * hours
        if state.fan_on:
            state.energy_used_wh += self.energy_fan_w * hours

        if actions:
            state.last_action_ts = now

        return state, actions
