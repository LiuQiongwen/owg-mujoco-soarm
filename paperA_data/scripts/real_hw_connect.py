"""
Phase 3 (real-robot validation, see /home/lina/.claude/plans/floating-crunching-yeti.md):
reusable connection helper for the physical SO-ARM101.

Works around two environment issues discovered 2026-07-10 (not hardware
problems):
  1. This project's own datasets/episode.py collides with the HuggingFace
     `datasets` package lerobot needs internally -- import lerobot's feetech
     submodule FIRST (while only site-packages is on sys.path), then manually
     register the project's datasets/episode.py under sys.modules['datasets.episode']
     before adding the project root to sys.path.
  2. `scservo_sdk` (the feetech-servo-sdk PyPI package) was not installed in
     the tango env -- installed via `pip install "feetech-servo-sdk>=1.0.0,<2.0.0"`
     (the exact extras pin from /lena/projects/lerobot/pyproject.toml).

Calibration file: found at the default lerobot user-cache location
(~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json),
saved from an earlier calibration run in the separate `openvla` project --
this is a lerobot-level, not project-level, cache convention, so it applies
here unchanged.

Usage:
    conda run -n tango python paperA_data/scripts/real_hw_connect.py
"""
import sys
import importlib.util
import numpy as np

PROJECT_ROOT = "/lena/projects/OWG-main"
CALIBRATION_PATH = (
    "/home/lina/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json"
)
PORT = "/dev/ttyACM0"
MAX_RELATIVE_TARGET_DEG = 30.0  # safety clamp: max joint delta per command


def _fix_datasets_collision():
    """Must run before PROJECT_ROOT is added to sys.path -- see module docstring."""
    from lerobot.motors.feetech import FeetechMotorsBus  # noqa: F401 (forces datasets resolution)
    spec = importlib.util.spec_from_file_location(
        "datasets.episode", f"{PROJECT_ROOT}/datasets/episode.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["datasets.episode"] = mod
    spec.loader.exec_module(mod)


def connect_backend(max_relative_target=MAX_RELATIVE_TARGET_DEG):
    """Returns a connected SOARMRealBackend. Caller is responsible for
    calling .close() when done (or use as a context manager)."""
    _fix_datasets_collision()
    sys.path.insert(0, PROJECT_ROOT)
    from robots.soarm_real_backend import SOARMRealBackend
    from cameras.base import CameraBase

    class _NullCamera(CameraBase):
        """No camera needed for pure arm connectivity/motion tests."""
        def camera_id(self): return "null"
        @property
        def width(self): return 0
        @property
        def height(self): return 0
        def fov_deg(self): return 0.0
        def intrinsics(self): return np.eye(3)
        def capture(self): raise NotImplementedError

    backend = SOARMRealBackend(
        port=PORT, camera=_NullCamera(),
        calibration=CALIBRATION_PATH,
        max_relative_target=max_relative_target,
    )
    backend.connect()
    return backend


if __name__ == "__main__":
    backend = connect_backend()
    try:
        q = backend.get_joint_positions()
        g = backend.get_gripper_opening()
        print("Connected OK.")
        print("Joint positions (rad):", q)
        print("Joint positions (deg):", np.degrees(q))
        print("Gripper opening (m):", g)
    finally:
        backend.close()
        print("Disconnected cleanly.")
