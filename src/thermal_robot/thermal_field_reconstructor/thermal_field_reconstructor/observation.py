"""Thermal observation contract and the top-down simulation projector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MEASUREMENT_FIELD_DIRECT = "field_direct"
MEASUREMENT_SURFACE_RADIANCE = "surface_radiance"


@dataclass(frozen=True)
class SensorPose2D:
    x: float
    y: float
    yaw: float
    frame_id: str = "world"


@dataclass
class ThermalObservation:
    stamp_s: float
    sensor_pose: SensorPose2D
    measurement_type: str
    sample_wx: np.ndarray
    sample_wy: np.ndarray
    temperature: np.ndarray
    confidence: np.ndarray
    sample_wz: np.ndarray | None = None
    sample_range_m: np.ndarray | None = None

    @property
    def frame_id(self):
        return self.sensor_pose.frame_id

    def __post_init__(self):
        n = int(np.asarray(self.temperature).size)
        for name in ("sample_wx", "sample_wy", "confidence"):
            size = int(np.asarray(getattr(self, name)).size)
            if size != n:
                raise ValueError(f"{name} size {size} != temperature size {n}")


def project_pixels_to_world(
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    robot_wx: float,
    robot_wy: float,
    robot_yaw: float,
):
    """Project a rectified top-down thermal image footprint into world coordinates."""
    px_xs = np.linspace(-fov_x / 2.0, fov_x / 2.0, width, dtype=np.float32)
    px_ys = np.linspace(-fov_y / 2.0, fov_y / 2.0, height, dtype=np.float32)
    px_xx, px_yy = np.meshgrid(px_xs, px_ys)
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    world_xs = robot_wx + cos_y * px_xx - sin_y * px_yy
    world_ys = robot_wy + sin_y * px_xx + cos_y * px_yy
    return world_xs, world_ys


class TopDownRectProjector:
    """Ideal top-down rectangular footprint projector used by the A-level sim."""

    measurement_type = MEASUREMENT_FIELD_DIRECT

    def __init__(self, fov_x: float = 4.0, fov_y: float = 3.0):
        self.fov_x = float(fov_x)
        self.fov_y = float(fov_y)

    def project(self, image: np.ndarray, pose: SensorPose2D, stamp_s: float) -> ThermalObservation:
        img = np.asarray(image, dtype=np.float32)
        wxs, wys = project_pixels_to_world(
            img.shape[1], img.shape[0], self.fov_x, self.fov_y,
            pose.x, pose.y, pose.yaw)
        temps = img.reshape(-1)
        return ThermalObservation(
            stamp_s=float(stamp_s),
            sensor_pose=pose,
            measurement_type=self.measurement_type,
            sample_wx=wxs.reshape(-1).astype(np.float32),
            sample_wy=wys.reshape(-1).astype(np.float32),
            temperature=temps,
            confidence=np.ones_like(temps, dtype=np.float32),
        )
