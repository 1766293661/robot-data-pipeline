from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawFrame:
    frame_index: int
    native_timestamp_sec: float | None
    values: dict[str, Any]
    camera_refs: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawEpisode:
    source_id: str
    source_revision: str
    native_episode_id: str
    source_format: str
    robot_type: str | None
    native_fps: float | None
    time_basis: str
    capabilities: dict[str, bool]
    task: dict[str, Any]
    locator: dict[str, Any]
    fingerprint: str
    frames: list[RawFrame]


@dataclass
class NormalizedFrame:
    frame_index: int
    native_timestamp_sec: float | None
    derived_timestamp_sec: float | None
    time_basis: str
    action: dict[str, Any] | None
    state: dict[str, Any] | None
    camera_refs: list[dict[str, Any]]
    extra: dict[str, Any]


@dataclass
class QualityResult:
    rule_id: str
    severity: str
    passed: bool
    evidence: dict[str, Any]
    rule_version: str = "1.0.0"
