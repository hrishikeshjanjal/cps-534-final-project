from __future__ import annotations

from typing import Dict, List


def generate_coaching_tip(sensor: Dict, actions: List[str]) -> str:
    """
    Lightweight rule-based 'LLM' that explains what the system did
    in natural language.
    """
    present = bool(sensor.get("present", False))
    light = float(sensor.get("light_lux", 0.0))
    temp = float(sensor.get("temperature_c", 0.0))
    humid = float(sensor.get("humidity_pct", 0.0))
    posture = float(sensor.get("posture_score", 1.0))

    if not actions:
        if not present:
            return "You seem away from your desk, so I'm keeping everything off to save energy."
        return "Conditions look comfortable, so I didn't change anything."

    # Build a short explanation based on actions
    parts: List[str] = []

    if "turn_on_light" in actions:
        parts.append(
            f"I turned on the desk light because it was quite dim (~{light:.0f} lux) while you were present."
        )
    if "turn_off_light" in actions:
        if present:
            parts.append(
                "I turned off the desk light because it was bright enough without extra lighting."
            )
        else:
            parts.append(
                "I turned off the desk light because you are not at the desk, to save energy."
            )

    if "turn_on_fan" in actions:
        parts.append(
            f"I turned on the fan as the environment felt warm/humid (≈{temp:.1f}°C, {humid:.0f}% RH)."
        )
    if "turn_off_fan" in actions:
        parts.append(
            "I turned off the fan because the temperature and humidity are back in a comfortable range."
        )

    if "posture_alert_on" in actions:
        parts.append(
            f"I sent a posture reminder since your posture score dropped to {posture:.2f}. Try sitting upright and relaxing your shoulders."
        )
    if "clear_posture_alert" in actions:
        parts.append(
            "Nice! Your posture looks better now, so I'm clearing the posture alert."
        )

    if not parts:
        # Fallback
        return "I adjusted your environment for comfort and energy efficiency."

    # Join into one sentence block
    return " ".join(parts)
