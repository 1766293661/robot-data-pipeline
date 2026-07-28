from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from core.config import SourceConfig
from core.models import RawEpisode


class Adapter(ABC):
    def __init__(self, source: SourceConfig):
        self.source = source

    @abstractmethod
    def discover(self, stored_episode_ids: set[str] | None = None) -> Iterable[RawEpisode]:
        """Yield source episodes without writing storage or applying quality decisions."""


def artifact_fingerprint(paths: list[Path]) -> str:
    """Cheap change detector for local input artifacts, not cross-source deduplication."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.name).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def stored_placeholder(source: SourceConfig, native_episode_id: str) -> RawEpisode:
    """Identity-only record sufficient for the pipeline to skip an already stored episode."""
    return RawEpisode(source.source_id, source.source_revision, native_episode_id, source.format, source.robot_type,
                      source.native_fps, source.time_basis, {}, {}, {}, "stored-placeholder", [])


def make_adapter(source: SourceConfig) -> Adapter:
    from .hdf5 import RobomimicHDF5Adapter
    from .lerobot import LeRobotAdapter
    from .oxe import OXETarAdapter

    adapters = {
        "lerobot": LeRobotAdapter,
        "oxe_tar": OXETarAdapter,
        "robomimic_hdf5": RobomimicHDF5Adapter,
    }
    try:
        return adapters[source.format](source)
    except KeyError as exc:
        raise ValueError(f"No adapter implemented for format {source.format!r}") from exc
