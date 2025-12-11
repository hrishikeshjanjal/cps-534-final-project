from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "llama3.1"
    max_tokens: int = 120
    temperature: float = 0.2
    endpoint: str = "http://localhost:11434/api/generate"


class LLMClient:
    """
    Lightweight LLM wrapper with an Ollama backend and rule-based fallback.
    """

    def __init__(self, config: Dict[str, Any], http_timeout: float = 8.0) -> None:
        llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
        self.config = LLMConfig(
            enabled=bool(llm_cfg.get("enabled", False)),
            provider=str(llm_cfg.get("provider", "ollama") or "ollama"),
            model=str(llm_cfg.get("model", "llama3.1") or "llama3.1"),
            max_tokens=int(llm_cfg.get("max_tokens", 120) or 0),
            temperature=float(llm_cfg.get("temperature", 0.2) or 0.0),
            endpoint=str(
                llm_cfg.get("endpoint", "http://localhost:11434/api/generate")
                or "http://localhost:11434/api/generate"
            ),
        )
        self.http_timeout = http_timeout

    def explain_decision(self, sensor: Dict[str, Any], actions: List[str]) -> str:
        """
        Provide a friendly sentence explaining why the actions make sense.
        Uses the LLM when enabled, otherwise falls back to the rule-based explainer.
        """
        prompt = self._build_explain_prompt(sensor, actions)
        llm_text = self._call_llm(prompt)
        if llm_text:
            return llm_text
        return self._rule_based_explanation(sensor, actions)

    def generate_coaching_tip(self, sensor: Dict[str, Any], actions: List[str]) -> str:
        """
        Provide a short ergonomic/comfort coaching tip.
        """
        prompt = self._build_coaching_prompt(sensor, actions)
        llm_text = self._call_llm(prompt)
        if llm_text:
            return llm_text
        return self._rule_based_coaching(sensor, actions)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call the Ollama endpoint. Returns None on failure to trigger fallback logic.
        """
        if not self.config.enabled or self.config.provider.lower() != "ollama":
            return None

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "options": {"temperature": self.config.temperature},
            "stream": False,
        }

        if self.config.max_tokens > 0:
            payload["options"]["num_predict"] = int(self.config.max_tokens)

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.http_timeout) as resp:
                if resp.status != 200:
                    return None
                raw = resp.read()
        except (error.URLError, TimeoutError):
            return None

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

        text = parsed.get("response")
        if isinstance(text, str):
            return text.strip()
        return None

    def _build_explain_prompt(
        self, sensor: Dict[str, Any], actions: List[str]
    ) -> str:
        ts = sensor.get("timestamp")
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        present = "present" if bool(sensor.get("present", False)) else "away"
        prompt = (
            "You are a smart desk assistant that adjusts light, fan, and posture alerts.\n"
            f"Timestamp: {ts_str}\n"
            f"Presence: {present}\n"
            f"Light lux: {float(sensor.get('light_lux', 0.0)):.1f}\n"
            f"Temperature C: {float(sensor.get('temperature_c', 0.0)):.1f}\n"
            f"Humidity %: {float(sensor.get('humidity_pct', 0.0)):.1f}\n"
            f"Posture score: {float(sensor.get('posture_score', 1.0)):.2f}\n"
            f"Actions taken: {', '.join(actions) if actions else 'none'}\n"
            "Respond with one friendly sentence explaining why these actions make sense "
            "from the perspective of a helpful desk assistant."
        )
        return prompt

    def _build_coaching_prompt(
        self, sensor: Dict[str, Any], actions: List[str]
    ) -> str:
        prompt = (
            "You are a smart desk assistant. Provide a short ergonomic or comfort tip "
            "(1-2 sentences) based on the posture and environment.\n"
            f"Presence: {'present' if bool(sensor.get('present', False)) else 'away'}\n"
            f"Light lux: {float(sensor.get('light_lux', 0.0)):.1f}\n"
            f"Temperature C: {float(sensor.get('temperature_c', 0.0)):.1f}\n"
            f"Humidity %: {float(sensor.get('humidity_pct', 0.0)):.1f}\n"
            f"Posture score: {float(sensor.get('posture_score', 1.0)):.2f}\n"
            f"Recent actions: {', '.join(actions) if actions else 'none'}\n"
            "Keep it actionable and supportive."
        )
        return prompt

    def _rule_based_explanation(
        self, sensor: Dict[str, Any], actions: List[str]
    ) -> str:
        present = bool(sensor.get("present", False))
        light = float(sensor.get("light_lux", 0.0))
        temp = float(sensor.get("temperature_c", 0.0))
        humid = float(sensor.get("humidity_pct", 0.0))
        posture = float(sensor.get("posture_score", 1.0))

        if not actions:
            if not present:
                return "No adjustments needed while you are away; keeping devices off to save energy."
            return "Everything looks comfortable right now, so I left the setup as-is."

        parts: List[str] = []

        if "turn_on_light" in actions:
            if present:
                parts.append(
                    f"I turned on the desk light because it was dim (~{light:.0f} lux) while you were here."
                )
            else:
                parts.append(
                    "I switched on the desk light so the space is ready even though you just stepped away."
                )
        if "turn_off_light" in actions:
            if present:
                parts.append(
                    "I turned off the desk light because the area was already bright enough."
                )
            else:
                parts.append(
                    "I turned off the desk light because you are not at the desk, to save energy."
                )

        if "turn_on_fan" in actions:
            parts.append(
                f"I turned on the fan since it felt warm or humid (about {temp:.1f}°C, {humid:.0f}% humidity)."
            )
        if "turn_off_fan" in actions:
            parts.append(
                "I turned off the fan because temperature and humidity returned to a comfortable range."
            )

        if "posture_alert_on" in actions:
            parts.append(
                f"I sent a posture reminder because your posture score dropped to {posture:.2f}. Please sit upright and relax your shoulders."
            )
        if "clear_posture_alert" in actions:
            parts.append(
                "Great job improving your posture — I'm clearing the alert for now."
            )

        if not parts:
            return "I adjusted your environment for comfort, efficiency, and better posture."

        return " ".join(parts)

    def _rule_based_coaching(
        self, sensor: Dict[str, Any], actions: List[str]
    ) -> str:
        temp = float(sensor.get("temperature_c", 0.0))
        humid = float(sensor.get("humidity_pct", 0.0))
        posture = float(sensor.get("posture_score", 1.0))

        if "posture_alert_on" in actions:
            return (
                "Try planting your feet flat, rolling your shoulders back, and keeping your screen at eye level."
            )
        if posture < 0.8:
            return "Take a quick stretch break and realign your spine to keep posture scores healthy."
        if "turn_on_fan" in actions or temp > 26 or humid > 65:
            return "Stay cool with steady airflow and sip some water to offset the warm, humid air."
        if "turn_on_light" in actions:
            return "Adjust your monitor brightness to match the desk light and reduce eye strain."

        return "Looks good — keep a relaxed, upright posture and take short breaks to stay comfortable."
