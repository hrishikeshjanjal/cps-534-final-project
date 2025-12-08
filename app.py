import streamlit as st
import pandas as pd

from src.eval.metrics import load_log, compute_metrics


def load_log_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    st.set_page_config(page_title="Smart Desk Assistant – Demo", layout="wide")
    st.title("🧠💡 Smart Desk Assistant – Demo")

    st.sidebar.header("Controls")
    log_path = st.sidebar.text_input(
        "Log file path",
        value="logs/simulation_log.csv",
        help="Path to a CSV log produced by run_simulation.py or scenarios.py",
    )

    if st.sidebar.button("Load log"):
        st.session_state["log_loaded"] = True
        st.session_state["log_path"] = log_path

    if "log_loaded" not in st.session_state or not st.session_state["log_loaded"]:
        st.info("Use the sidebar to load a log file (e.g. logs/simulation_log.csv).")
        return

    path = st.session_state["log_path"]
    try:
        df = load_log_df(path)
    except Exception as e:
        st.error(f"Failed to load log: {e}")
        return

    # Metrics
    metrics = compute_metrics(df)

    st.subheader("Metrics Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total steps", metrics["total_steps"])
        st.metric("Decisions", metrics["num_decisions"])
    with col2:
        st.metric("Avg latency (ms)", f"{metrics['avg_latency_ms']:.4f}")
        st.metric("P95 latency (ms)", f"{metrics['p95_latency_ms']:.4f}")
    with col3:
        st.metric("Light on ratio", f"{metrics['light_on_ratio']:.2f}")
        st.metric("Energy used (Wh)", f"{metrics['total_energy_wh']:.2f}")

    st.markdown("---")

    # Presence + light
    st.subheader("Presence & Light Over Time")
    df_pl = df[["timestamp", "present", "light_lux"]].copy()
    df_pl = df_pl.set_index("timestamp")
    st.line_chart(df_pl)

    st.markdown("---")

    # Temperature + humidity
    st.subheader("Temperature & Humidity Over Time")
    df_th = df[["timestamp", "temperature_c", "humidity_pct"]].copy()
    df_th = df_th.set_index("timestamp")
    st.line_chart(df_th)

    st.markdown("---")

    # Decisions and LLM messages
    st.subheader("Decision Timeline & Assistant Messages")
    decisions = df[df["actions"].fillna("") != ""].copy()
    if not decisions.empty:
        decisions_view = decisions[
            ["timestamp", "present", "light_lux", "temperature_c", "humidity_pct", "actions", "llm_message"]
        ]
        st.dataframe(decisions_view)
    else:
        st.write("No actions were taken during this run.")


if __name__ == "__main__":
    main()
