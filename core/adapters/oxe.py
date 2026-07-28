from __future__ import annotations

import pickle
import tarfile
from pathlib import Path
from typing import Any, Iterable

from core.adapters.base import Adapter, artifact_fingerprint, relative_posix, stored_placeholder
from core.models import RawEpisode, RawFrame


class OXETarAdapter(Adapter):
    """Read a trusted local OXE/WebDataset tar export one pickle member at a time."""

    def discover(self, stored_episode_ids: set[str] | None = None) -> Iterable[RawEpisode]:
        stored_episode_ids = stored_episode_ids or set()
        root = self.source.root
        tar_files = sorted(root.glob(self.source.adapter_options.get("tar_glob", "*.tar")))
        if not tar_files:
            raise FileNotFoundError(f"No OXE tar files found under {root}")
        for tar_path in tar_files:
            fingerprint = artifact_fingerprint([tar_path])
            with tarfile.open(tar_path) as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".data.pickle"):
                        continue
                    native_episode_id = member.name.removesuffix(".data.pickle")
                    if native_episode_id in stored_episode_ids:
                        yield stored_placeholder(self.source, native_episode_id)
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    payload = pickle.load(stream)  # Input is a user-provided, trusted local public dataset.
                    steps = payload.get("steps", [])
                    frames: list[RawFrame] = []
                    for index, step in enumerate(steps):
                        observation = step.get("observation", {})
                        image = observation.get("image")
                        camera_refs = []
                        if isinstance(image, (bytes, bytearray)):
                            camera_refs.append({
                                "kind": "embedded_tar_pickle",
                                "container_relative_path": relative_posix(root, tar_path),
                                "member_name": member.name,
                                "step_index": index,
                                "field_path": f"steps[{index}].observation.image",
                                "encoding": "jpeg",
                            })
                        instruction = observation.get("natural_language_instruction")
                        if isinstance(instruction, bytes):
                            instruction = instruction.decode("utf-8", errors="replace")
                        extra = {
                            "reward": _scalar(step.get("reward")),
                            "termination": {"is_first": bool(step.get("is_first", False)), "is_last": bool(step.get("is_last", False)), "is_terminal": bool(step.get("is_terminal", False))},
                            "language_instruction": instruction,
                            "unknown_field_names": [key for key in step if key not in {"action", "observation", "reward", "is_first", "is_last", "is_terminal"}],
                        }
                        frames.append(RawFrame(index, None, {"action": step.get("action"), "observation": observation}, camera_refs, extra))
                    capabilities = {"action": True, "state": bool(steps and "state" in steps[0].get("observation", {})), "proprioception": bool(steps and "state" in steps[0].get("observation", {})), "images": bool(steps and isinstance(steps[0].get("observation", {}).get("image"), (bytes, bytearray))), "language_instruction": bool(steps and steps[0].get("observation", {}).get("natural_language_instruction")), "robot_identity": bool(self.source.robot_type)}
                    locator = {"container_relative_path": relative_posix(root, tar_path), "member_name": member.name}
                    yield RawEpisode(self.source.source_id, self.source.source_revision, native_episode_id, "oxe_tar", self.source.robot_type,
                                     self.source.native_fps, self.source.time_basis, capabilities, {"text": None, "native_task_id": None}, locator, fingerprint, frames)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
