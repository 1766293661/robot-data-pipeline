from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from core.store import Store


def export_jsonl(database_path: Path, output_path: Path, frame_budget: int, clip_length: int, include_needs_review: bool = False) -> dict[str, Any]:
    if frame_budget < 1 or clip_length < 1:
        raise ValueError("frame_budget and clip_length must be positive")
    store = Store(database_path)
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for episode in store.accepted_episode_rows(include_needs_review):
        frames = store.load_frames(int(episode["episode_pk"]))
        for start in range(0, len(frames), clip_length):
            buckets[(episode["source_id"], episode["robot_type"] or "unknown")].append({"episode": episode, "frames": frames[start:start + clip_length]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected, per_bucket = 0, defaultdict(int)
    with output_path.open("w", encoding="utf-8") as handle:
        while buckets and selected < frame_budget:
            for bucket in list(buckets):
                if selected >= frame_budget:
                    break
                clip = buckets[bucket].popleft()
                frames = clip["frames"][:frame_budget - selected]
                if frames:
                    handle.write(json.dumps(_record(clip["episode"], frames), ensure_ascii=False) + "\n")
                    selected += len(frames)
                    per_bucket["/".join(bucket)] += len(frames)
                if not buckets[bucket]:
                    del buckets[bucket]
    store.close()
    return {"output": str(output_path), "frames": selected, "by_source_robot": dict(per_bucket)}


def _record(episode, frames) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "source_id": episode["source_id"],
        "source_revision": episode["source_revision"],
        "robot_type": episode["robot_type"],
        "native_episode_id": episode["native_episode_id"],
        "frame_range": {"frame_start": frames[0].frame_index, "frame_end": frames[-1].frame_index},
        "time_basis": frames[0].time_basis,
        "quality_status": episode["quality_status"],
        "frames": [{
            "frame_index": frame.frame_index,
            "native_timestamp_sec": frame.native_timestamp_sec,
            "derived_timestamp_sec": frame.derived_timestamp_sec,
            "action": frame.action,
            "state": frame.state,
            "camera_refs": frame.camera_refs,
            "extra": frame.extra,
        } for frame in frames],
    }
