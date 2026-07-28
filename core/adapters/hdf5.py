from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py

from core.adapters.base import Adapter, artifact_fingerprint, relative_posix, stored_placeholder
from core.models import RawEpisode, RawFrame


class RobomimicHDF5Adapter(Adapter):
    def discover(self, stored_episode_ids: set[str] | None = None) -> Iterable[RawEpisode]:
        stored_episode_ids = stored_episode_ids or set()
        root = self.source.root
        file_name = self.source.adapter_options.get("file", "test.hdf5")
        hdf5_path = root / file_name
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Robomimic HDF5 file not found: {hdf5_path}")
        options = self.source.adapter_options
        fingerprint = artifact_fingerprint([hdf5_path])
        with h5py.File(hdf5_path, "r") as handle:
            demos = handle[options["demo_root"]]
            for native_episode_id in sorted(demos):
                if native_episode_id in stored_episode_ids:
                    yield stored_placeholder(self.source, native_episode_id)
                    continue
                demo = demos[native_episode_id]
                actions = demo[options["action_dataset"]]
                states = demo[options["state_dataset"]]
                dones = demo.get(options.get("done_dataset", "dones"))
                rewards = demo.get(options.get("reward_dataset", "rewards"))
                camera_names = options.get("camera_datasets", [])
                frames: list[RawFrame] = []
                for index in range(len(actions)):
                    camera_refs = [{
                        "kind": "hdf5_dataset",
                        "container_relative_path": relative_posix(root, hdf5_path),
                        "dataset_path": f"{options['demo_root']}/{native_episode_id}/{options['observation_root']}/{camera_name}",
                        "frame_index": index,
                        "encoding": "rgb_uint8",
                    } for camera_name in camera_names if f"{options['observation_root']}/{camera_name}" in demo]
                    extra = {
                        "reward": _to_python(rewards[index]) if rewards is not None else None,
                        "termination": {"done": bool(dones[index]) if dones is not None else False},
                        "unknown_field_names": [],
                    }
                    frames.append(RawFrame(index, None, {"actions": actions[index], "states": states[index]}, camera_refs, extra))
                capabilities = {"action": True, "state": True, "proprioception": True, "images": bool(camera_names), "language_instruction": False, "robot_identity": bool(self.source.robot_type)}
                locator = {"container_relative_path": relative_posix(root, hdf5_path), "demo_path": f"{options['demo_root']}/{native_episode_id}"}
                yield RawEpisode(self.source.source_id, self.source.source_revision, native_episode_id, "robomimic_hdf5", self.source.robot_type,
                                 self.source.native_fps, self.source.time_basis, capabilities, {"text": None, "native_task_id": native_episode_id}, locator, fingerprint, frames)


def _to_python(value):
    return value.item() if hasattr(value, "item") else value
