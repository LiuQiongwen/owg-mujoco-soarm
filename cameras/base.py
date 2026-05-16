"""Abstract camera interface shared by simulated and real cameras."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RGBDFrame:
    """Single camera observation snapshot."""
    rgb:       np.ndarray          # (H, W, 3) uint8
    depth:     np.ndarray          # (H, W) float32 in metres
    timestamp: float = field(default_factory=time.time)
    camera_id: str   = "overhead"

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


class CameraBase(ABC):
    """Minimal interface every camera backend must implement."""

    @property
    @abstractmethod
    def camera_id(self) -> str: ...

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @property
    @abstractmethod
    def fov_deg(self) -> float: ...

    @property
    @abstractmethod
    def intrinsics(self) -> np.ndarray:
        """3×3 camera intrinsics matrix K."""
        ...

    @abstractmethod
    def capture(self) -> RGBDFrame:
        """Capture one RGB-D frame. Must be non-blocking."""
        ...

    def capture_rgb(self) -> np.ndarray:
        return self.capture().rgb

    def capture_depth(self) -> np.ndarray:
        return self.capture().depth
