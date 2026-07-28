from __future__ import annotations

from copy import deepcopy
from typing import Any


# Profiles contain versioned, source-layout-specific defaults. A source config may
# override any nested field when it encounters a nonstandard export.
SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "lerobot_aloha_dual_arm": {
        "format": "lerobot",
        "robot_type": "dual_arm",
        "native_fps": 50,
        "time_basis": "native_timestamp",
        "action_mapping": {
            "source_field_path": "action",
            "representation": "joint_position_target",
            "units": "rad",
            "coordinate_frame": "joint_space",
            "groups": {"left_arm": [0, 1, 2, 3, 4, 5, 6], "right_arm": [7, 8, 9, 10, 11, 12, 13]},
        },
        "state_mapping": {
            "source_field_path": "observation.state",
            "representation": "joint_position",
            "units": "rad",
            "coordinate_frame": "joint_space",
            "groups": {"left_arm": [0, 1, 2, 3, 4, 5, 6], "right_arm": [7, 8, 9, 10, 11, 12, 13]},
        },
    },
    "oxe_bridge": {
        "format": "oxe_tar",
        "robot_type": "single_arm",
        "time_basis": "step_index",
        "action_mapping": {
            "source_field_path": "action",
            "representation": "composite_cartesian_delta_and_gripper",
            "units": "mixed",
            "coordinate_frame": "mixed",
            "components": {
                "world_vector": {"source_field_path": "action.world_vector", "representation": "cartesian_translation_delta", "units": "m", "coordinate_frame": "world"},
                "rotation_delta": {"source_field_path": "action.rotation_delta", "representation": "rotation_delta", "units": "rad", "coordinate_frame": "dataset_native"},
                "open_gripper": {"source_field_path": "action.open_gripper", "representation": "gripper_open_command", "units": "boolean"},
                "terminate_episode": {"source_field_path": "action.terminate_episode", "representation": "termination_command", "units": "boolean"},
            },
        },
        "state_mapping": {"source_field_path": "observation.state", "representation": "dataset_native_state", "units": "dataset_native"},
        "adapter_options": {"image_field_path": "steps[].observation.image"},
    },
    "robomimic_panda_low_dim": {
        "format": "robomimic_hdf5",
        "robot_type": "panda_single_arm",
        "time_basis": "step_index",
        "action_mapping": {"source_field_path": "actions", "representation": "dataset_native_action", "units": "dataset_native", "coordinate_frame": "unspecified"},
        "state_mapping": {"source_field_path": "states", "representation": "mujoco_simulator_state", "units": "dataset_native", "coordinate_frame": "simulation_world"},
        "adapter_options": {
            "file": "test.hdf5", "demo_root": "data", "action_dataset": "actions", "state_dataset": "states",
            "observation_root": "obs", "done_dataset": "dones", "reward_dataset": "rewards",
            "camera_datasets": ["agentview_image", "robot0_eye_in_hand_image"],
        },
    },
}


def apply_profile(source: dict[str, Any]) -> dict[str, Any]:
    profile_name = source.get("profile")
    if not profile_name:
        return source
    try:
        defaults = deepcopy(SOURCE_PROFILES[profile_name])
    except KeyError as exc:
        available = ", ".join(sorted(SOURCE_PROFILES))
        raise ValueError(f"unknown source profile {profile_name!r}; available: {available}") from exc
    explicit_format = source.get("format")
    if explicit_format and explicit_format != defaults["format"]:
        raise ValueError(f"profile {profile_name!r} requires format {defaults['format']!r}")
    return _deep_merge(defaults, source)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
