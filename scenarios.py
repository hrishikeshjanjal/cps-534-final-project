import argparse
import time

from src.config_loader import load_config
from src.core.state import DeskState
from src.core.controller import Controller
from src.ai.llm_client import generate_coaching_tip
from src.hw.actuators import ConsoleActuator
from src.logging.logger import SimulationLogger
from src.data.scenarios import (
    generate_scenario_a,
    generate_scenario_b,
    generate_scenario_c,
    save_scenario_to_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run predefined Smart Desk scenarios")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["A", "B", "C"],
        help="Scenario ID: A (dim light), B (hot & humid), C (bad posture)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/scenario_log.csv",
        help="Where to store scenario CSV log",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="",
        help="Optional: save the scenario input itself to this CSV path",
    )
    return parser.parse_args()


def build_scenario_df(scenario_id: str, config):
    sampling_cfg = config.get("sampling", {})
    dt = int(sampling_cfg.get("period_seconds", 10))
    duration = int(sampling_cfg.get("duration_seconds", 600))

    if scenario_id == "A":
        return generate_scenario_a(duration_seconds=duration, sample_period_seconds=dt)
    if scenario_id == "B":
        return generate_scenario_b(duration_seconds=duration, sample_period_seconds=dt)
    if scenario_id == "C":
        return generate_scenario_c(duration_seconds=duration, sample_period_seconds=dt)
    raise ValueError(f"Unknown scenario: {scenario_id}")


def main():
    args = parse_args()
    config = load_config(args.config)

    controller = Controller(config)
    state = DeskState()
    actuator = ConsoleActuator()
    logger = SimulationLogger(args.log_path)

    df = build_scenario_df(args.scenario, config)

    if args.save_csv:
        save_scenario_to_csv(df, args.save_csv)

    print(f"Running Scenario {args.scenario} with {len(df)} steps...")

    try:
        last_ts = None
        for ts, row in df.iterrows():
            sensor = {
                "timestamp": ts.to_pydatetime(),
                "present": bool(row["present"]),
                "distance_cm": float(row["distance_cm"]),
                "light_lux": float(row["light_lux"]),
                "temperature_c": float(row["temperature_c"]),
                "humidity_pct": float(row["humidity_pct"]),
                "posture_score": float(row["posture_score"]),
            }

            if last_ts is None:
                delta = config.get("sampling", {}).get("period_seconds", 10)
            else:
                delta = (ts - last_ts).total_seconds()
            last_ts = ts

            t0 = time.perf_counter()
            state, actions = controller.step(sensor, state, dt_seconds=delta, now=ts)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            llm_message = ""
            if actions:
                llm_message = generate_coaching_tip(sensor, actions)
                print(
                    f"[{ts.isoformat()}] present={sensor['present']} "
                    f"light={sensor['light_lux']:.1f} temp={sensor['temperature_c']:.1f}C "
                    f"hum={sensor['humidity_pct']:.1f}% -> actions={actions}"
                )
                print(f"  [ASSISTANT] {llm_message}")

            actuator.set_light(state.light_on)
            actuator.set_fan(state.fan_on)
            if state.posture_alert_on:
                actuator.send_posture_notification(
                    "Your posture is degrading. Please sit upright and relax your shoulders."
                )

            logger.log_step(
                timestamp=ts,
                sensor=sensor,
                state=state,
                actions=actions,
                latency_ms=latency_ms,
                llm_message=llm_message,
            )
    finally:
        logger.close()

    print("\nScenario finished.")
    print(f"Total energy used (Wh): {state.energy_used_wh:.2f}")


if __name__ == "__main__":
    main()
