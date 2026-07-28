from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from core.profiles import apply_profile


class RuntimeConfig(BaseModel):
    batch_size: int = Field(default=512, ge=1)
    worker_count: int = Field(default=4, ge=1)
    max_in_flight_episodes: int = Field(default=8, ge=1)


class PathConfig(BaseModel):
    input_root: Path = Path("input_data")
    output_root: Path = Path("output")
    database_filename: str = "robot_pipeline.sqlite"

    @property
    def database_path(self) -> Path:
        return self.output_root / self.database_filename


class ExportConfig(BaseModel):
    frame_budget: int = Field(default=50_000, ge=1)
    clip_length: int = Field(default=16, ge=1)
    output_filename: str = "train.jsonl"
    include_needs_review: bool = False


class SignalComponentMapping(BaseModel):
    source_field_path: str
    representation: str
    units: str
    coordinate_frame: str | None = None


class SignalMapping(BaseModel):
    source_field_path: str
    representation: str
    units: str
    coordinate_frame: str | None = None
    groups: dict[str, list[int]] = Field(default_factory=dict)
    components: dict[str, SignalComponentMapping] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    source_id: str
    source_revision: str
    source_uri: str
    profile: str | None = None
    format: Literal["lerobot", "oxe_tar", "robomimic_hdf5"]
    root: Path
    robot_type: str | None = None
    native_fps: float | None = Field(default=None, gt=0)
    time_basis: Literal["native_timestamp", "derived_from_fps", "step_index"]
    action_mapping: SignalMapping | None = None
    state_mapping: SignalMapping | None = None
    adapter_options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def apply_source_profile(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return apply_profile(value)

    @field_validator("source_id", "source_revision")
    @classmethod
    def non_empty_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def require_fps_for_derived_time(self) -> "SourceConfig":
        if self.time_basis == "derived_from_fps" and self.native_fps is None:
            raise ValueError("native_fps is required when time_basis is derived_from_fps")
        return self


class PipelineConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    sources: list[SourceConfig]

    @model_validator(mode="after")
    def unique_source_versions(self) -> "PipelineConfig":
        identities = [(source.source_id, source.source_revision) for source in self.sources]
        if len(identities) != len(set(identities)):
            raise ValueError("each (source_id, source_revision) pair must appear once")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    """Load YAML and resolve all local paths against the configuration file."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = PipelineConfig.model_validate(raw)
    base = config_path.parent
    config.paths.input_root = _resolve(base, config.paths.input_root)
    config.paths.output_root = _resolve(base, config.paths.output_root)
    for source in config.sources:
        source.root = source.root if source.root.is_absolute() else (config.paths.input_root / source.root).resolve()
    return config


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()
