from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DeskState:
    light_on: bool = False
    fan_on: bool = False
    posture_alert_on: bool = False
    last_action_ts: Optional[datetime] = None
    energy_used_wh: float = 0.0
