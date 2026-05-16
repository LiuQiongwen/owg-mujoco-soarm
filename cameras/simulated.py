"""SimulatedCamera — wraps MuJoCo overhead renderer from EnvironmentSoArm."""

from __future__ import annotations

import math
import time

import numpy as np

from cameras.base import CameraBase, RGBDFrame


class SimulatedCamera(CameraBase):
    """
    Thin wrapper around EnvironmentSoArm's overhead renderer.

    The env already holds the MuJoCo Renderer and exposes
    _render_rgb_depth_seg().  This class presents the same CameraBase
    interface so robot code and data-collection scripts don't have to
    know whether they're talking to MuJoCo or real hardware.
    """

    def __init__(self, env, camera_name: str = "overhead"):
        self._env         = env
        self._camera_name = camera_name

    # ── CameraBase interface ──────────────────────────────────────────────────

    @property
    def camera_id(self) -> str:
        return self._camera_name

    @property
    def width(self) -> int:
        return self._env.camera.width

    @property
    def height(self) -> int:
        return self._env.camera.height

    @property
    def fov_deg(self) -> float:
        return self._env.camera.fov

    @property
    def intrinsics(self) -> np.ndarray:
        """3×3 intrinsics computed from image size and fov."""
        h, w   = self.height, self.width
        fy     = h / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        fx     = fy * (w / h)
        return np.array([
            [fx,  0.0, w / 2.0],
            [0.0,  fy, h / 2.0],
            [0.0, 0.0,     1.0],
        ], dtype=np.float32)

    def capture(self) -> RGBDFrame:
        rgb, depth, _seg = self._env._render_rgb_depth_seg()
        return RGBDFrame(
            rgb=rgb.astype(np.uint8),
            depth=depth.astype(np.float32),
            timestamp=time.time(),
            camera_id=self._camera_name,
        )

    # ── extras (sim-only) ─────────────────────────────────────────────────────

    def capture_with_seg(self):
        """Return (RGBDFrame, seg) — segmentation is MuJoCo-specific."""
        rgb, depth, seg = self._env._render_rgb_depth_seg()
        frame = RGBDFrame(
            rgb=rgb.astype(np.uint8),
            depth=depth.astype(np.float32),
            timestamp=time.time(),
            camera_id=self._camera_name,
        )
        return frame, seg
