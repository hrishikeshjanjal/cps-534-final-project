from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def analyze_video(
    video_path: str,
    sample_fps: int = 1,
    slouch_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Perform a lightweight brightness + posture heuristic over a video.

    - Samples frames at ~sample_fps.
    - Brightness: mean grayscale intensity (0-255).
    - Posture/slouch heuristic: vertical center of mass; lower values indicate slouching/crouching.
      This is intentionally simple and can be replaced by a real pose model later.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("opencv-python is required for video analysis.") from exc

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(int(round(fps / float(sample_fps))) if fps > 0 else 1, 1)

    brightness_vals = []
    slouch_flags = []
    frame_idx = 0
    sampled_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:  # sample every `stride` frames (0-based)
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        brightness_vals.append(brightness)
        slouch_flags.append(_estimate_slouch(gray, slouch_threshold))
        sampled_frames += 1
        frame_idx += 1

    cap.release()

    if not brightness_vals:
        return {
            "num_frames": 0,
            "sample_fps": sample_fps,
            "avg_brightness": 0.0,
            "min_brightness": 0.0,
            "max_brightness": 0.0,
            "estimated_slouch_ratio": 0.0,
        }

    avg_brightness = float(sum(brightness_vals) / len(brightness_vals))
    min_brightness = float(min(brightness_vals))
    max_brightness = float(max(brightness_vals))
    estimated_slouch_ratio = (
        float(sum(1 for f in slouch_flags if f)) / float(len(slouch_flags))
    )

    return {
        "num_frames": int(sampled_frames),
        "sample_fps": sample_fps,
        "avg_brightness": avg_brightness,
        "min_brightness": min_brightness,
        "max_brightness": max_brightness,
        "estimated_slouch_ratio": estimated_slouch_ratio,
    }


def _estimate_slouch(gray_frame, threshold: float) -> bool:
    """
    Simple heuristic: compute vertical center of mass of pixel intensities.
    If the center is lower than `threshold` fraction of the frame height, assume slouching.
    """
    import cv2  # type: ignore

    h, _w = gray_frame.shape[:2]

    # Use edges (subject contours) to find vertical center-of-mass;
    # fallback to grayscale intensities if edges are too sparse.
    blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    weights = edges.astype("float32")
    total = weights.sum()
    if total < 1.0:
        weights = blur.astype("float32")
        total = weights.sum()

    if total <= 0:
        return False

    row_indices = np.arange(h, dtype="float32").reshape(-1, 1)
    center_of_mass = float((weights * row_indices).sum() / total)
    ratio = center_of_mass / float(h)

    # Extra signal: if the lower half is much heavier than the upper, count as slouching.
    top = float(weights[: h // 2, :].sum()) + 1e-6
    bottom = float(weights[h // 2 :, :].sum()) + 1e-6
    lower_ratio = bottom / (top + bottom)
    bottom_heavy = lower_ratio > 0.62

    # Require some contrast to avoid flagging dark noisy frames
    edge_density = total / float(h * _w)

    return (ratio > threshold or bottom_heavy) and edge_density > 2.5
