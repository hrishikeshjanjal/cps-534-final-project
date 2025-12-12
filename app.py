import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.ai.llm_client import LLMClient
from src.config_loader import load_config
from src.eval.metrics import compute_metrics
from src.video.analyzer import analyze_video


def load_log_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def ensure_session_state() -> None:
    st.session_state.setdefault("log_loaded", False)
    st.session_state.setdefault("log_path", "logs/simulation_log.csv")


def render_metrics_tab() -> None:
    if not st.session_state.get("log_loaded"):
        st.info("Use the sidebar to load a log file (e.g. logs/simulation_log.csv).")
        return

    path = st.session_state.get("log_path", "logs/simulation_log.csv")
    try:
        df = load_log_df(path)
    except Exception as e:
        st.error(f"Failed to load log: {e}")
        return

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

    st.subheader("Presence & Light Over Time")
    df_pl = df[["timestamp", "present", "light_lux"]].copy().set_index("timestamp")
    st.line_chart(df_pl)

    st.markdown("---")

    st.subheader("Temperature & Humidity Over Time")
    df_th = df[["timestamp", "temperature_c", "humidity_pct"]].copy().set_index(
        "timestamp"
    )
    st.line_chart(df_th)

    st.markdown("---")

    st.subheader("Decision Timeline & Assistant Messages")
    decisions = df[df["actions"].fillna("") != ""].copy()
    if not decisions.empty:
        decisions_view = decisions[
            [
                "timestamp",
                "present",
                "light_lux",
                "temperature_c",
                "humidity_pct",
                "actions",
                "llm_message",
            ]
        ]
        st.dataframe(decisions_view)
    else:
        st.write("No actions were taken during this run.")


def render_video_feedback_tab(config_path: str) -> None:
    st.subheader("Video Feedback Demo")
    st.write(
        "Upload a short desk video (e.g., dim light or slouching) and let the Smart Desk Assistant "
        "summarize lighting/posture and give GenAI feedback."
    )

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )
    col1, col2 = st.columns(2)
    sample_fps = col1.slider("Sampling rate (frames/sec)", 1, 5, 1, step=1)
    slouch_threshold = col2.slider(
        "Slouch heuristic threshold",
        0.45,
        0.70,
        0.55,
        step=0.01,
        help="Higher values mark more frames as slouched.",
    )

    analyze_clicked = st.button(
        "Analyze Video", type="primary", disabled=uploaded is None
    )

    if not analyze_clicked:
        return

    if uploaded is None:
        st.warning("Please upload a video first.")
        return

    tmp_path = None
    try:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analyzing video..."):
            summary = analyze_video(
                tmp_path, sample_fps=sample_fps, slouch_threshold=slouch_threshold
            )

        if summary.get("num_frames", 0) == 0:
            st.warning("No frames were sampled from this video.")

        st.success("Analysis complete.")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Frames analyzed", summary.get("num_frames", 0))
        col_a.metric("Avg brightness (0-255)", f"{summary.get('avg_brightness', 0):.1f}")
        col_b.metric("Min brightness", f"{summary.get('min_brightness', 0):.1f}")
        col_b.metric("Max brightness", f"{summary.get('max_brightness', 0):.1f}")
        col_c.metric(
            "Estimated slouch ratio", f"{summary.get('estimated_slouch_ratio', 0):.2f}"
        )
        st.caption(
            "Brightness uses mean grayscale intensity; slouching uses an edge-based vertical center-of-mass heuristic."
        )

        thinking_box = st.empty()
        thinking_box.info("GenAI is thinking...")
        with st.spinner("Generating GenAI feedback..."):
            # Fake a short thinking delay so users see the stage even on fallback
            import time

            time.sleep(0.8)
            feedback, debug_log, llm_source = generate_video_feedback(
                summary=summary, video_name=uploaded.name, config_path=config_path
            )
        thinking_box.empty()

        st.markdown("**GenAI Feedback**")
        st.markdown(feedback or "No feedback available.")
        with st.expander("GenAI debug log"):
            for line in debug_log:
                st.write(line)

        log_video_feedback(
            summary=summary,
            video_name=uploaded.name,
            llm_feedback=feedback,
            llm_source=llm_source,
            log_path="logs/video_feedback.csv",
        )
    except Exception as exc:
        st.error(f"Video analysis failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def build_video_feedback_prompt(summary: Dict[str, Any], video_name: str) -> str:
    return (
        "You are the Smart Desk Assistant reviewing a desk video. "
        "Provide 2-3 friendly sentences that connect to your behavior "
        "(light control, posture alerts, comfort).\n"
        f"Video: {video_name}\n"
        f"Frames analyzed: {summary.get('num_frames', 0)} at ~{summary.get('sample_fps', 1)} fps\n"
        f"Brightness (avg/min/max on 0-255 scale): {summary.get('avg_brightness', 0):.1f} / "
        f"{summary.get('min_brightness', 0):.1f} / {summary.get('max_brightness', 0):.1f}\n"
        f"Estimated slouch ratio (0-1): {summary.get('estimated_slouch_ratio', 0):.2f}\n"
        "Comment on whether lighting seems dim/bright and whether posture needs reminders. "
        "Mention how you'd adjust the desk light or send posture alerts."
    )


def fallback_video_feedback(summary: Dict[str, Any]) -> str:
    brightness = float(summary.get("avg_brightness", 0.0))
    slouch_ratio = float(summary.get("estimated_slouch_ratio", 0.0))

    if brightness < 60:
        light_msg = "desk is very dim, so I'd bring lights up for visibility"
    elif brightness < 120:
        light_msg = "desk looks somewhat dim; a desk light bump would help"
    elif brightness > 200:
        light_msg = "desk is quite bright; we could dim or rely on ambient light"
    else:
        light_msg = "lighting is balanced, no major changes needed"

    if slouch_ratio > 0.6:
        posture_msg = "saw frequent slouching and would keep posture alerts active"
    elif slouch_ratio > 0.3:
        posture_msg = "noticed occasional slouching, so light posture nudges help"
    else:
        posture_msg = "posture looked steady for most of the clip"

    return (
        f"I analyzed the clip: average brightness ~{brightness:.1f}/255 and slouch ratio {slouch_ratio:.2f}. "
        f"The {light_msg}, and I {posture_msg} to keep you comfortable."
    )


def generate_video_feedback(
    summary: Dict[str, Any], video_name: str, config_path: str
) -> (str, list[str], str):
    """
    Returns (feedback_text, debug_log_lines, llm_source)
    llm_source is 'llm' when Ollama returns text, otherwise 'fallback'.
    """
    try:
        config = load_config(config_path)
    except Exception:
        config = {}
    llm_client = LLMClient(config)
    prompt = build_video_feedback_prompt(summary, video_name)
    fallback = fallback_video_feedback(summary)

    llm_text, reason = llm_client.call_with_reason(prompt)
    source = "llm" if llm_text else "fallback"
    feedback = llm_text if llm_text else fallback

    debug_log = [
        f"LLM enabled: {config.get('llm', {}).get('enabled', False)}",
        f"LLM provider: {config.get('llm', {}).get('provider', 'ollama')}",
        f"LLM model: {config.get('llm', {}).get('model', 'llama3.1')}",
        f"Source used: {source}",
        f"LLM reason: {reason}",
        "Prompt:",
        prompt,
        "Feedback:",
        feedback,
    ]
    return feedback, debug_log, source


def log_video_feedback(
    summary: Dict[str, Any],
    video_name: str,
    llm_feedback: str,
    llm_source: str,
    log_path: str,
) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "timestamp",
        "video_name",
        "num_frames",
        "sample_fps",
        "avg_brightness",
        "min_brightness",
        "max_brightness",
        "estimated_slouch_ratio",
        "llm_feedback",
        "llm_source",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "video_name": video_name,
                "num_frames": summary.get("num_frames", 0),
                "sample_fps": summary.get("sample_fps", 0),
                "avg_brightness": summary.get("avg_brightness", 0.0),
                "min_brightness": summary.get("min_brightness", 0.0),
                "max_brightness": summary.get("max_brightness", 0.0),
                "estimated_slouch_ratio": summary.get("estimated_slouch_ratio", 0.0),
                "llm_feedback": llm_feedback,
                "llm_source": llm_source,
            }
        )


def main():
    st.set_page_config(page_title="Smart Desk Assistant – Demo", layout="wide")
    ensure_session_state()
    st.title("🧠💡 Smart Desk Assistant – Demo")

    st.sidebar.header("Controls")
    log_path = st.sidebar.text_input(
        "Log file path",
        value=st.session_state.get("log_path", "logs/simulation_log.csv"),
        help="Path to a CSV log produced by run_simulation.py or scenarios.py",
    )
    config_path = st.sidebar.text_input(
        "Config file path",
        value="configs/config.yaml",
        help="Used for LLM settings in the video feedback demo.",
    )

    if st.sidebar.button("Load log"):
        st.session_state["log_loaded"] = True
        st.session_state["log_path"] = log_path

    tab_metrics, tab_video = st.tabs(["Simulation Metrics", "Video Feedback Demo"])
    with tab_metrics:
        render_metrics_tab()
    with tab_video:
        render_video_feedback_tab(config_path)


if __name__ == "__main__":
    main()
