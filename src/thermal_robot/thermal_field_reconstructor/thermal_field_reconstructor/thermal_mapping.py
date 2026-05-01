"""World-frame thermal map fusion primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GridSnapshot:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    temperature_mean: np.ndarray
    temperature_variance: np.ndarray
    confidence: np.ndarray
    visit_count: np.ndarray
    last_seen_age_s: np.ndarray


def project_pixels_to_world(
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    robot_wx: float,
    robot_wy: float,
    robot_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a rectified thermal image footprint into world coordinates."""
    px_xs = np.linspace(-fov_x / 2.0, fov_x / 2.0, width, dtype=np.float32)
    px_ys = np.linspace(-fov_y / 2.0, fov_y / 2.0, height, dtype=np.float32)
    px_xx, px_yy = np.meshgrid(px_xs, px_ys)
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    world_xs = robot_wx + cos_y * px_xx - sin_y * px_yy
    world_ys = robot_wy + sin_y * px_xx + cos_y * px_yy
    return world_xs, world_ys


class WorldThermalGrid:
    """Fuses thermal image observations into a world-frame grid."""

    def __init__(
        self,
        center_x: float = -6.0,
        center_y: float = 0.0,
        size_x_m: float = 50.0,
        size_y_m: float = 50.0,
        resolution: float = 0.25,
        ambient_temp: float = 22.0,
        confidence_visit_scale: float = 6.0,
        age_decay_s: float = 45.0,
        unknown_variance: float = 100.0,
    ):
        self.resolution = float(resolution)
        self.width = int(round(size_x_m / self.resolution))
        self.height = int(round(size_y_m / self.resolution))
        self.origin_x = float(center_x - size_x_m / 2.0)
        self.origin_y = float(center_y - size_y_m / 2.0)
        shape = (self.height, self.width)
        self.mean = np.full(shape, ambient_temp, dtype=np.float32)
        self._m2 = np.zeros(shape, dtype=np.float32)
        self.visit_count = np.zeros(shape, dtype=np.uint32)
        self.last_seen = np.full(shape, -1.0, dtype=np.float32)
        self.ambient_temp = float(ambient_temp)
        self.confidence_visit_scale = max(1.0, float(confidence_visit_scale))
        self.age_decay_s = max(1e-3, float(age_decay_s))
        self.unknown_variance = float(unknown_variance)

    def world_to_cell(self, wx: np.ndarray, wy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((wx - self.origin_x) / self.resolution).astype(np.int32)
        iy = np.floor((wy - self.origin_y) / self.resolution).astype(np.int32)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        return ix, iy, valid

    def cell_to_world(self, ix: np.ndarray, iy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        wx = self.origin_x + (ix.astype(np.float32) + 0.5) * self.resolution
        wy = self.origin_y + (iy.astype(np.float32) + 0.5) * self.resolution
        return wx, wy

    def integrate_image(
        self,
        image: np.ndarray,
        robot_wx: float,
        robot_wy: float,
        robot_yaw: float,
        stamp_s: float,
        fov_x: float = 4.0,
        fov_y: float = 3.0,
    ) -> None:
        world_xs, world_ys = project_pixels_to_world(
            image.shape[1], image.shape[0], fov_x, fov_y, robot_wx, robot_wy, robot_yaw
        )
        ix, iy, valid = self.world_to_cell(world_xs.reshape(-1), world_ys.reshape(-1))
        if not np.any(valid):
            return
        values = image.reshape(-1).astype(np.float32)[valid]
        linear = iy[valid] * self.width + ix[valid]
        total_cells = self.width * self.height
        obs_count = np.bincount(linear, minlength=total_cells).astype(np.float32)
        obs_sum = np.bincount(linear, weights=values, minlength=total_cells).astype(np.float32)
        obs_sum_sq = np.bincount(linear, weights=values * values, minlength=total_cells).astype(np.float32)
        cells = np.flatnonzero(obs_count > 0.0)
        if cells.size == 0:
            return

        obs_n = obs_count[cells]
        obs_mean = obs_sum[cells] / obs_n
        obs_m2 = np.maximum(0.0, obs_sum_sq[cells] - obs_n * obs_mean * obs_mean)

        mean_flat = self.mean.reshape(-1)
        m2_flat = self._m2.reshape(-1)
        visit_flat = self.visit_count.reshape(-1)
        last_flat = self.last_seen.reshape(-1)

        prev_n = visit_flat[cells].astype(np.float32)
        prev_mean = mean_flat[cells]
        new_n = prev_n + obs_n
        delta = obs_mean - prev_mean
        mean_flat[cells] = prev_mean + delta * obs_n / np.maximum(new_n, 1.0)
        m2_flat[cells] = m2_flat[cells] + obs_m2 + delta * delta * prev_n * obs_n / np.maximum(new_n, 1.0)
        visit_flat[cells] = np.clip(new_n, 0, np.iinfo(np.uint32).max).astype(np.uint32)
        last_flat[cells] = float(stamp_s)

    def snapshot(self, now_s: float) -> GridSnapshot:
        count_f = self.visit_count.astype(np.float32)
        variance = np.full_like(self.mean, self.unknown_variance, dtype=np.float32)
        seen = self.visit_count > 1
        variance[seen] = self._m2[seen] / np.maximum(count_f[seen] - 1.0, 1.0)
        visited = self.last_seen >= 0.0
        age = np.full_like(self.mean, -1.0, dtype=np.float32)
        age[visited] = np.maximum(0.0, float(now_s) - self.last_seen[visited])
        confidence = np.zeros_like(self.mean, dtype=np.float32)
        confidence[visited] = np.minimum(1.0, count_f[visited] / self.confidence_visit_scale)
        confidence[visited] *= np.exp(-age[visited] / self.age_decay_s).astype(np.float32)
        return GridSnapshot(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            temperature_mean=self.mean.copy(),
            temperature_variance=variance,
            confidence=confidence,
            visit_count=self.visit_count.copy(),
            last_seen_age_s=age,
        )
