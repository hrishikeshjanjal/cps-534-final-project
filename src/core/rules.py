from __future__ import annotations


def decide_light_action(
    light_lux: float,
    present: bool,
    on_threshold: float,
    off_threshold: float,
) -> str:
    if not present:
        return "turn_off_light"

    if light_lux < on_threshold:
        return "turn_on_light"
    if light_lux > off_threshold:
        return "turn_off_light"
    return "no_change"


def decide_fan_action(
    temperature_c: float,
    humidity_pct: float,
    present: bool,
    temp_on: float,
    temp_off: float,
    humid_on: float,
) -> str:
    if not present:
        return "turn_off_fan"

    if temperature_c > temp_on or humidity_pct > humid_on:
        return "turn_on_fan"
    if temperature_c < temp_off:
        return "turn_off_fan"
    return "no_change"


def decide_posture_alert(
    posture_score: float,
    present: bool,
    bad_posture_threshold: float,
) -> str:
    if not present:
        return "clear_posture_alert"

    if posture_score < bad_posture_threshold:
        return "posture_alert_on"
    return "clear_posture_alert"
