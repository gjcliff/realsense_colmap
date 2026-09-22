"""Pinhole camera intrinsics, shared between the RealSense capture and the
COLMAP / depth-fusion stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float  # meters per raw depth unit

    def as_matrix(self):
        import numpy as np

        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def colmap_params(self) -> str:
        """Comma-separated params for COLMAP's PINHOLE camera model."""
        return f"{self.fx},{self.fy},{self.cx},{self.cy}"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Intrinsics":
        return cls(**json.loads(path.read_text()))


@dataclass
class ColorCalibration:
    """Color sensor intrinsics plus its extrinsic offset from the
    depth/infrared camera frame (which is what everything else in this
    project treats as "the" camera). Used only to reproject the RGB image
    onto geometry computed from depth+IR, never fed into COLMAP -- the color
    sensor is rolling-shutter and never drives pose estimation here."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: list[float]  # row-major 3x3: p_color = rotation @ p_depth + translation
    translation: list[float]  # meters

    def as_matrix(self):
        import numpy as np

        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def rotation_matrix(self):
        import numpy as np

        return np.array(self.rotation, dtype=np.float64).reshape(3, 3)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ColorCalibration":
        return cls(**json.loads(path.read_text()))
