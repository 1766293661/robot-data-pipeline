from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.config import SignalComponentMapping, SignalMapping, SourceConfig
from core.models import NormalizedFrame, RawEpisode


def normalize_episode(raw: RawEpisode, source: SourceConfig) -> list[NormalizedFrame]:
    frames: list[NormalizedFrame] = []
    for frame in raw.frames:
        native_timestamp = frame.native_timestamp_sec if source.time_basis == "native_timestamp" else None
        derived_timestamp = frame.frame_index / source.native_fps if source.time_basis == "derived_from_fps" and source.native_fps else None
        frames.append(NormalizedFrame(
            frame_index=frame.frame_index,
            native_timestamp_sec=native_timestamp,
            derived_timestamp_sec=derived_timestamp,
            time_basis=source.time_basis,
            action=_normalize_signal(frame.values, source.action_mapping),
            state=_normalize_signal(frame.values, source.state_mapping),
            camera_refs=_jsonable(frame.camera_refs),
            extra=_jsonable(frame.extra),
        ))
    return frames


def _normalize_signal(values: dict[str, Any], mapping: SignalMapping | None) -> dict[str, Any] | None:
    if mapping is None:
        return None
    if mapping.components:
        return {
            "form": "named_components", "source_field_path": mapping.source_field_path,
            "representation": mapping.representation, "values": None, "dimension": None,
            "units": mapping.units, "coordinate_frame": mapping.coordinate_frame, "groups": None,
            "components": {name: _normalize_component(values, component) for name, component in mapping.components.items()},
        }
    vector = _numeric_vector(_resolve_field(values, mapping.source_field_path))
    return {
        "form": "vector", "source_field_path": mapping.source_field_path,
        "representation": mapping.representation, "values": vector, "dimension": len(vector),
        "units": mapping.units, "coordinate_frame": mapping.coordinate_frame,
        "groups": mapping.groups, "components": None,
    }


def _normalize_component(values: dict[str, Any], component: SignalComponentMapping) -> dict[str, Any]:
    raw = _resolve_field(values, component.source_field_path)
    vector = _numeric_vector(raw)
    result: dict[str, Any] = {
        "source_field_path": component.source_field_path,
        "representation": component.representation,
        "units": component.units,
        "coordinate_frame": component.coordinate_frame,
    }
    if len(vector) == 1:
        result["value"] = vector[0]
    else:
        result["values"] = vector
    return result


def _resolve_field(values: dict[str, Any], path: str) -> Any:
    """Resolve literal Parquet keys first, then dotted nested mappings."""
    if path in values:
        return values[path]
    current: Any = values
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (bool, np.bool_)):
        return [float(value)]
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, dict):
        return []
    if hasattr(value, "tolist"):
        return _numeric_vector(value.tolist())
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_numeric_vector(item))
        return flattened
    return []


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if isinstance(value, bytes):
        return {"encoding": "bytes", "length": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return repr(value)
