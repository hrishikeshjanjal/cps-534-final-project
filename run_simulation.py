import argparse
import time

from src.config_loader import load_config
from src.data.stream import get_sensor_stream
from src.core.state import DeskState
from src.core.controller import Controller
from src.ai.llm_client import generate_coaching_tip
from src.hw.actuators import ConsoleActuator
from src.logging.logger import SimulationLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Desk Assistant simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/simulation_log.csv",
        help="Where to store simulation CSV log",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    controller = Controller(config)
    state = DeskState()

    sampling_cfg = config.get("sampling", {})
    dt_default = int(sampling_cfg.get("period_seconds", 10))

    stream = get_sensor_stream(config)

    actuator = ConsoleActuator()
    logger = SimulationLogger(args.log_path)

    last_ts = None
    print("Starting simulation with Berkeley + HAR data...")

    try:
        for i, sensor in enumerate(stream, start=1):
            ts = sensor["timestamp"]
            if last_ts is None:
                delta = dt_default
            else:
                delta = (ts - last_ts).total_seconds()
            last_ts = ts

            t0 = time.perf_counter()
            state, actions = controller.step(sensor, state, dt_seconds=delta, now=ts)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # LLM-style explanation
            llm_message = ""
            if actions:
                llm_message = generate_coaching_tip(sensor, actions)
                print(
                    f"[{ts.isoformat()}] present={sensor['present']} "
                    f"light={sensor['light_lux']:.1f} temp={sensor['temperature_c']:.1f}C "
                    f"hum={sensor['humidity_pct']:.1f}% -> actions={actions}"
                )
                print(f"  [ASSISTANT] {llm_message}")

            # Actuators (simulated)
            actuator.set_light(state.light_on)
            actuator.set_fan(state.fan_on)
            if state.posture_alert_on:
                actuator.send_posture_notification(
                    "Your posture is degrading. Please sit upright and relax your shoulders."
                )

            # Log this step
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

    print("\nSimulation finished.")
    print(f"Total energy used (Wh): {state.energy_used_wh:.2f}")


if __name__ == "__main__":
    main()
