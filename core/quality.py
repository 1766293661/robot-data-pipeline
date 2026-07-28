from __future__ import annotations

import math
from typing import Any

from core.models import NormalizedFrame, QualityResult


def evaluate_episode(frames: list[NormalizedFrame], native_fps: float | None, capabilities: dict[str, bool]) -> tuple[str, list[QualityResult]]:
    results = [
        _episode_length(frames),
        _declared_signal_presence(frames, capabilities),
        _timestamps(frames, native_fps),
        _action_finite_and_range(frames),
        _action_jump(frames),
        _state_action_identical(frames),
        _camera_references(frames, capabilities),
    ]
    if any(not result.passed and result.severity == "error" for result in results):
        return "rejected", results
    if any(not result.passed for result in results):
        return "needs_review", results
    return "accepted", results


def _episode_length(frames: list[NormalizedFrame]) -> QualityResult:
    return QualityResult("episode_minimum_length", "warning", len(frames) >= 3, {"frame_count": len(frames), "minimum": 3})


def _declared_signal_presence(frames: list[NormalizedFrame], capabilities: dict[str, bool]) -> QualityResult:
    missing: dict[str, list[int]] = {}
    for name in ("action", "state"):
        if not capabilities.get(name, False):
            continue
        frame_indexes = [frame.frame_index for frame in frames if not _signal_values(getattr(frame, name))]
        if frame_indexes:
            missing[name] = frame_indexes[:20]
    return QualityResult("declared_signal_presence", "error", not missing, {"missing_frame_indexes": missing})


def _timestamps(frames: list[NormalizedFrame], native_fps: float | None) -> QualityResult:
    timestamps = [frame.native_timestamp_sec for frame in frames if frame.native_timestamp_sec is not None]
    if not timestamps:
        return QualityResult("timestamp_monotonicity", "warning", True, {"applicable": False, "reason": "no_native_timestamp"})
    monotonic = all(right > left for left, right in zip(timestamps, timestamps[1:]))
    evidence: dict[str, Any] = {"applicable": True, "monotonic": monotonic}
    interval_ok = True
    if native_fps and len(timestamps) > 1:
        expected = 1 / native_fps
        errors = [abs((right - left) - expected) / expected for left, right in zip(timestamps, timestamps[1:])]
        evidence["max_relative_interval_error"] = max(errors)
        interval_ok = evidence["max_relative_interval_error"] <= 0.25
    return QualityResult("timestamp_monotonicity", "error", monotonic and interval_ok, evidence)


def _action_finite_and_range(frames: list[NormalizedFrame]) -> QualityResult:
    values = [value for frame in frames for value in _signal_values(frame.action)]
    if not values:
        return QualityResult("action_finite_and_range", "warning", True, {"applicable": False, "reason": "no_action"})
    finite = all(math.isfinite(value) for value in values)
    maximum = max((abs(value) for value in values), default=0.0)
    return QualityResult("action_finite_and_range", "error", finite and maximum <= 10.0, {"max_absolute_value": maximum, "limit": 10.0})


def _action_jump(frames: list[NormalizedFrame]) -> QualityResult:
    vectors = [_signal_values(frame.action) for frame in frames]
    jumps = []
    for left, right in zip(vectors, vectors[1:]):
        if left and len(left) == len(right):
            jumps.append(max(abs(a - b) for a, b in zip(left, right)))
    if not jumps:
        return QualityResult("action_frame_jump", "warning", True, {"applicable": False})
    maximum = max(jumps)
    return QualityResult("action_frame_jump", "error", maximum <= 5.0, {"max_jump": maximum, "limit": 5.0})


def _state_action_identical(frames: list[NormalizedFrame]) -> QualityResult:
    comparable = []
    for frame in frames:
        action, state = _signal_values(frame.action), _signal_values(frame.state)
        if action and len(action) == len(state):
            comparable.append(action == state)
    if not comparable:
        return QualityResult("state_action_identity", "warning", True, {"applicable": False})
    ratio = sum(comparable) / len(comparable)
    return QualityResult("state_action_identity", "warning", ratio < 0.98, {"identical_ratio": ratio})


def _camera_references(frames: list[NormalizedFrame], capabilities: dict[str, bool]) -> QualityResult:
    if not capabilities.get("images", False):
        return QualityResult("camera_reference_completeness", "warning", True, {"applicable": False})
    missing = [frame.frame_index for frame in frames if not frame.camera_refs]
    malformed = []
    for frame in frames:
        for reference in frame.camera_refs:
            if not reference.get("kind") or not reference.get("encoding"):
                malformed.append(frame.frame_index)
            if reference.get("kind") == "lerobot_video" and reference.get("video_frame_index") is None and reference.get("pts") is None:
                malformed.append(frame.frame_index)
    return QualityResult("camera_reference_completeness", "warning", not missing and not malformed, {"missing_frames": missing[:20], "malformed_frames": malformed[:20]})


def _signal_values(signal: dict[str, Any] | None) -> list[float]:
    if signal is None:
        return []
    if signal.get("form") == "vector":
        return [float(value) for value in signal.get("values", [])]
    values: list[float] = []
    for component in signal.get("components", {}).values():
        if "value" in component:
            values.append(float(component["value"]))
        else:
            values.extend(float(value) for value in component.get("values", []))
    return values
