"""RealSenseCamera stub — placeholder for Intel RealSense D435i integration.

Import this only on hardware that has pyrealsense2 installed.
The stub raises ImportError with a helpful message if pyrealsense2 is absent
so that simulation code continues to work on machines without the SDK.

Usage (real hardware):
    from cameras.realsense_stub import RealSenseCamera
    cam = RealSenseCamera(width=640, height=480, fps=30)
    frame = cam.capture()
    print(frame.rgb.shape, frame.depth.shape)
"""

from __future__ import annotations

import time

import numpy as np

from cameras.base import CameraBase, RGBDFrame

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False


class RealSenseCamera(CameraBase):
    """
    Intel RealSense D435i camera.

    Aligns depth to color frame on construction.
    Depth pixels are in metres (float32).

    Raises
    ------
    ImportError
        If pyrealsense2 is not installed.
    RuntimeError
        If no RealSense device is found.
    """

    def __init__(
        self,
        width:     int   = 640,
        height:    int   = 480,
        fps:       int   = 30,
        serial:    str   = "",        # device serial — empty = first device
        camera_id: str   = "realsense",
    ):
        if not _RS_AVAILABLE:
            raise ImportError(
                "pyrealsense2 is not installed. "
                "Install it with: pip install pyrealsense2"
            )

        self._width     = width
        self._height    = height
        self._fps       = fps
        self._camera_id = camera_id

        self._pipe    = rs.pipeline()
        cfg           = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)

        profile        = self._pipe.start(cfg)
        depth_sensor   = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        align_to       = rs.stream.color
        self._align    = rs.align(align_to)

        # Intrinsics from the device
        color_stream   = profile.get_stream(rs.stream.color)
        intr           = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics_mat = np.array([
            [intr.fx,    0.0, intr.ppx],
            [0.0,    intr.fy, intr.ppy],
            [0.0,       0.0,       1.0],
        ], dtype=np.float32)
        self._fov_deg  = float(
            2.0 * np.degrees(np.arctan(height / (2.0 * intr.fy)))
        )

    # ── CameraBase interface ──────────────────────────────────────────────────

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fov_deg(self) -> float:
        return self._fov_deg

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics_mat.copy()

    def capture(self) -> RGBDFrame:
        frames        = self._pipe.wait_for_frames()
        aligned       = self._align.process(frames)
        color_frame   = aligned.get_color_frame()
        depth_frame   = aligned.get_depth_frame()

        rgb   = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        raw_d = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        depth = raw_d.astype(np.float32) * self._depth_scale   # metres

        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            timestamp=time.time(),
            camera_id=self._camera_id,
        )

    def close(self):
        self._pipe.stop()

    def __del__(self):
        try:
            self._pipe.stop()
        except Exception:
            pass
