"""Load and fingerprint versioned Piper embodiment metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import yaml

from .contracts import stable_hash


@dataclass(frozen=True)
class EmbodimentMetadata:
    values: Mapping[str, Any]
    config_hash: str


def load_embodiment_metadata(path: str | Path) -> EmbodimentMetadata:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("embodiment metadata must be a mapping")
    if not values.get("schema_version") or not values.get("model_variant_id"):
        raise ValueError("schema_version and model_variant_id are required")
    if values.get("status") != "pre-freeze":
        raise ValueError("this integration package only loads pre-freeze metadata")
    return EmbodimentMetadata(values=values, config_hash=stable_hash(values))


def validate_metadata_against_assets(metadata: EmbodimentMetadata, repo_root: str | Path = ".") -> None:
    """Fail if the manifest drifts from the compile-time MJCF assets."""
    root = Path(repo_root)
    values = metadata.values
    arm_xml = ET.parse(root / values["source_assets"]["arm"]).getroot()
    grip_xml = ET.parse(root / values["source_assets"]["gripper"]).getroot()

    joint_order = values["arm"]["joint_order"]
    expected_limits = values["arm"]["joint_limits_rad"]
    joints = {element.attrib.get("name"): element for element in arm_xml.iter("joint")}
    actual_limits = [list(map(float, joints[name].attrib["range"].split())) for name in joint_order]
    if actual_limits != expected_limits:
        raise ValueError("arm joint limits drifted from embodiment metadata")

    expected_ctrl = list(map(float, values["gripper"]["actuator_control_range_m"]))
    actuators = [
        element for element in grip_xml.iter("position")
        if element.attrib.get("name") == "gripper_finger_joint7"
    ]
    if len(actuators) != 1 or list(map(float, actuators[0].attrib["ctrlrange"].split())) != expected_ctrl:
        raise ValueError("gripper control range drifted from embodiment metadata")

    geom_names = {element.attrib.get("name") for element in grip_xml.iter("geom")}
    if not set(values["gripper"]["collision_geoms"]).issubset(geom_names):
        raise ValueError("finger collision metadata drifted from gripper asset")
