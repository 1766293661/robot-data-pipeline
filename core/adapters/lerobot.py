from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import av
import pyarrow.parquet as pq

from core.adapters.base import Adapter, artifact_fingerprint, relative_posix, stored_placeholder
from core.models import RawEpisode, RawFrame


class LeRobotAdapter(Adapter):
    """Read standard LeRobot v2/v3 layouts, including the flat local slice layout."""

    def __init__(self, source):
        super().__init__(source)
        self._pts_cache: dict[Path, tuple[list[int], str, str]] = {}

    def discover(self, stored_episode_ids: set[str] | None = None) -> Iterable[RawEpisode]:
        stored_episode_ids = stored_episode_ids or set()
        root = self.source.root
        info = self._load_info(root)
        data_files = self._data_files(root)
        if not data_files:
            raise FileNotFoundError(f"No LeRobot data parquet files found under {root}")
        episode_meta = self._episode_metadata(root)
        if episode_meta and set(episode_meta).issubset(stored_episode_ids):
            for native_episode_id in sorted(episode_meta, key=int):
                yield stored_placeholder(self.source, native_episode_id)
            return
        video = self._find_video(root)
        video_pts: list[int] | None = None
        time_base: str | None = None
        codec: str | None = None
        fingerprint = artifact_fingerprint(data_files + ([video] if video else []))
        rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for data_file in data_files:
            parquet = pq.ParquetFile(data_file)
            for batch in parquet.iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    rows_by_episode[str(row["episode_index"])].append(row)

        feature_info = info.get("features", {})
        camera_keys = [key for key, value in feature_info.items() if isinstance(value, dict) and value.get("dtype") == "video"]
        declared_video_codec = next((value.get("video_info", {}).get("video.codec") for value in feature_info.values()
                                     if isinstance(value, dict) and value.get("dtype") == "video" and value.get("video_info", {}).get("video.codec")), None)
        robot_type = self.source.robot_type or {"aloha": "dual_arm"}.get(info.get("robot_type"), info.get("robot_type"))
        fps = self.source.native_fps or info.get("fps")
        for native_episode_id, rows in sorted(rows_by_episode.items(), key=lambda item: int(item[0])):
            if native_episode_id in stored_episode_ids:
                yield stored_placeholder(self.source, native_episode_id)
                continue
            if video is not None and video_pts is None:
                video_pts, time_base, codec = self._video_pts(video)
            rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0))))
            meta = episode_meta.get(native_episode_id, {})
            frames: list[RawFrame] = []
            for local_index, row in enumerate(rows):
                frame_index = int(row.get("frame_index", local_index))
                timestamp = _as_float(row.get("timestamp"))
                camera_refs = self._camera_refs(
                    root=root,
                    video=video,
                    video_pts=video_pts or [],
                    time_base=time_base,
                    codec=declared_video_codec or codec,
                    camera_keys=camera_keys,
                    global_index=int(row.get("index", frame_index)),
                    timestamp=timestamp,
                    episode_meta=meta,
                )
                extra = {
                    "termination": {"done": bool(row.get("next.done", False))},
                    "task_index": row.get("task_index"),
                    "unknown_field_names": [key for key in row if key not in {"action", "observation.state", "episode_index", "frame_index", "timestamp", "index", "next.done", "task_index"}],
                }
                frames.append(RawFrame(frame_index, timestamp, row, camera_refs, extra))
            task = {"text": (meta.get("tasks") or [None])[0], "native_task_id": rows[0].get("task_index")}
            capabilities = {"action": True, "state": True, "proprioception": True, "images": bool(video), "language_instruction": False, "robot_identity": True}
            locator = {"data_files": [relative_posix(root, path) for path in data_files], "episode_metadata": meta}
            yield RawEpisode(self.source.source_id, self.source.source_revision, native_episode_id, "lerobot", robot_type,
                             float(fps) if fps else None, self.source.time_basis, capabilities, task, locator, fingerprint, frames)

    @staticmethod
    def _load_info(root: Path) -> dict[str, Any]:
        for path in (root / "meta" / "info.json", root / "info.json"):
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _data_files(root: Path) -> list[Path]:
        flat = sorted(root.glob("file-*.parquet"))
        if flat:
            return flat
        data_root = root / "data"
        files = sorted(data_root.rglob("*.parquet")) if data_root.exists() else []
        return [path for path in files if path.name not in {"episodes.parquet", "tasks.parquet"}]

    @staticmethod
    def _episode_metadata(root: Path) -> dict[str, dict[str, Any]]:
        for path in (root / "meta" / "episodes.parquet", root / "episodes.parquet"):
            if path.exists():
                return {str(row["episode_index"]): row for row in pq.read_table(path).to_pylist()}
        return {}

    @staticmethod
    def _find_video(root: Path) -> Path | None:
        direct = sorted(root.glob("*.mp4"))
        if direct:
            return direct[0]
        videos = root / "videos"
        matches = sorted(videos.rglob("*.mp4")) if videos.exists() else []
        return matches[0] if matches else None

    def _video_pts(self, path: Path) -> tuple[list[int], str, str]:
        cached = self._pts_cache.get(path)
        if cached is not None:
            return cached
        with av.open(path) as container:
            stream = container.streams.video[0]
            time_base = str(stream.time_base)
            codec = stream.codec_context.name
            pts = [int(frame.pts) for frame in container.decode(stream) if frame.pts is not None]
        self._pts_cache[path] = (pts, time_base, codec)
        return pts, time_base, codec

    @staticmethod
    def _camera_refs(*, root: Path, video: Path | None, video_pts: list[int], time_base: str | None,
                     codec: str | None, camera_keys: list[str], global_index: int, timestamp: float | None, episode_meta: dict[str, Any]) -> list[dict[str, Any]]:
        if video is None:
            return []
        if 0 <= global_index < len(video_pts):
            video_frame_index: int | None = global_index
            pts: int | None = video_pts[global_index]
        else:
            video_frame_index, pts = None, None
        start = episode_meta.get("videos/observation.images.top/from_timestamp")
        end = episode_meta.get("videos/observation.images.top/to_timestamp")
        return [{
            "kind": "lerobot_video",
            "camera_key": key,
            "video_relative_path": relative_posix(root, video),
            "video_frame_index": video_frame_index,
            "pts": pts,
            "pts_time_base": time_base,
            "alignment_method": "global_dataset_index_to_decoded_video_frame",
            "native_timestamp_sec": timestamp,
            "episode_timestamp_start_sec": _as_float(start),
            "episode_timestamp_end_sec": _as_float(end),
            "encoding": codec or "video",
        } for key in camera_keys or ["default"]]


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None
