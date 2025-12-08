from __future__ import annotations

from abc import ABC, abstractmethod


class DeskActuator(ABC):
    @abstractmethod
    def set_light(self, on: bool) -> None:
        ...

    @abstractmethod
    def set_fan(self, on: bool) -> None:
        ...

    @abstractmethod
    def send_posture_notification(self, message: str) -> None:
        ...


class ConsoleActuator(DeskActuator):
    """
    Simple actuator that just prints what it would do.
    For a real deployment this could talk to GPIO / MQTT / etc.
    """

    def __init__(self) -> None:
        self._light_state = None
        self._fan_state = None
        self._posture_active = False

    def set_light(self, on: bool) -> None:
        if self._light_state is None or self._light_state != on:
            state_str = "ON" if on else "OFF"
            print(f"  [ACTUATOR] Desk light -> {state_str}")
            self._light_state = on

    def set_fan(self, on: bool) -> None:
        if self._fan_state is None or self._fan_state != on:
            state_str = "ON" if on else "OFF"
            print(f"  [ACTUATOR] Fan -> {state_str}")
            self._fan_state = on

    def send_posture_notification(self, message: str) -> None:
        # Only print when posture alerts toggle on
        if not self._posture_active:
            print(f"  [ACTUATOR] Posture alert: {message}")
            self._posture_active = True
        # Clearing posture is done implicitly when controller turns posture_alert_off
        # (we don't need to spam the console)
