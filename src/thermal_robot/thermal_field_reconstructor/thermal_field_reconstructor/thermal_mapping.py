"""World-frame thermal map fusion primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from thermal_field_reconstructor import visibility as _visibility
from thermal_field_reconstructor.grid_geometry import (
    GRID_CENTER_X, GRID_CENTER_Y, GRID_SIZE_M)
from thermal_field_reconstructor.observation import (  # noqa: F401
    SensorPose2D, ThermalObservation, TopDownRectProjector,
    project_pixels_to_world)

VIEW_NEVER = 0
VIEW_BLOCKED_ONLY = 1
VIEW_CLEAR = 2
N_VIEW_SECTORS = 8


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
    view_state: np.ndarray
    view_sectors: np.ndarray


class WorldThermalGrid:
    """Fuses thermal image observations into a world-frame grid."""

    def __init__(
        self,
        center_x: float = GRID_CENTER_X,
        center_y: float = GRID_CENTER_Y,
        size_x_m: float = GRID_SIZE_M,
        size_y_m: float = GRID_SIZE_M,
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
        self.blocked_count = np.zeros(shape, dtype=np.uint32)
        self.view_sectors = np.zeros(shape, dtype=np.uint8)

    def world_to_cell(self, wx: np.ndarray, wy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((wx - self.origin_x) / self.resolution).astype(np.int32)
        iy = np.floor((wy - self.origin_y) / self.resolution).astype(np.int32)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        return ix, iy, valid

    def cell_to_world(self, ix: np.ndarray, iy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        wx = self.origin_x + (ix.astype(np.float32) + 0.5) * self.resolution
        wy = self.origin_y + (iy.astype(np.float32) + 0.5) * self.resolution
        return wx, wy

    def integrate_observation(self, obs, occupancy=None,
                              ray_step_m=_visibility.DEFAULT_RAY_STEP_M) -> None:
        """Fuse contract observations, integrating only cells visible in occupancy."""
        ix, iy, valid = self.world_to_cell(obs.sample_wx, obs.sample_wy)
        if not np.any(valid):
            return
        values = np.asarray(obs.temperature, dtype=np.float32)[valid]
        linear = iy[valid] * self.width + ix[valid]
        total_cells = self.width * self.height
        obs_count = np.bincount(linear, minlength=total_cells).astype(np.float32)
        obs_sum = np.bincount(linear, weights=values, minlength=total_cells).astype(np.float32)
        obs_sum_sq = np.bincount(linear, weights=values * values, minlength=total_cells).astype(np.float32)
        cells = np.flatnonzero(obs_count > 0.0)
        if cells.size == 0:
            return

        cell_ix = (cells % self.width).astype(np.int32)
        cell_iy = (cells // self.width).astype(np.int32)
        cwx, cwy = self.cell_to_world(cell_ix, cell_iy)
        if occupancy is not None:
            visible = _visibility.visible_mask(
                occupancy, obs.sensor_pose.x, obs.sensor_pose.y,
                cwx, cwy, step_m=ray_step_m)
        else:
            visible = np.ones(cells.size, dtype=bool)

        blocked_cells = cells[~visible]
        if blocked_cells.size:
            blocked_flat = self.blocked_count.reshape(-1)
            blocked_flat[blocked_cells] += 1

        clear = cells[visible]
        if clear.size == 0:
            return
        obs_n = obs_count[clear]
        obs_mean = obs_sum[clear] / obs_n
        obs_m2 = np.maximum(0.0, obs_sum_sq[clear] - obs_n * obs_mean * obs_mean)

        mean_flat = self.mean.reshape(-1)
        m2_flat = self._m2.reshape(-1)
        visit_flat = self.visit_count.reshape(-1)
        last_flat = self.last_seen.reshape(-1)

        prev_n = visit_flat[clear].astype(np.float32)
        prev_mean = mean_flat[clear]
        new_n = prev_n + obs_n
        delta = obs_mean - prev_mean
        mean_flat[clear] = prev_mean + delta * obs_n / np.maximum(new_n, 1.0)
        m2_flat[clear] = m2_flat[clear] + obs_m2 + delta * delta * prev_n * obs_n / np.maximum(new_n, 1.0)
        visit_flat[clear] = np.clip(new_n, 0, np.iinfo(np.uint32).max).astype(np.uint32)
        last_flat[clear] = float(obs.stamp_s)

        sector_flat = self.view_sectors.reshape(-1)
        az = np.arctan2(obs.sensor_pose.y - cwy[visible],
                        obs.sensor_pose.x - cwx[visible])
        sector = (((az + np.pi) / (2.0 * np.pi)) * N_VIEW_SECTORS).astype(np.int32)
        sector = np.clip(sector, 0, N_VIEW_SECTORS - 1)
        sector_flat[clear] |= (1 << sector).astype(np.uint8)

    def integrate_image(
        self,
        image: np.ndarray,
        robot_wx: float,
        robot_wy: float,
        robot_yaw: float,
        stamp_s: float,
        fov_x: float = 4.0,
        fov_y: float = 3.0,
        occupancy=None,
    ) -> None:
        projector = TopDownRectProjector(fov_x=fov_x, fov_y=fov_y)
        obs = projector.project(
            np.asarray(image, dtype=np.float32),
            SensorPose2D(x=float(robot_wx), y=float(robot_wy), yaw=float(robot_yaw)),
            float(stamp_s))
        self.integrate_observation(obs, occupancy=occupancy)

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
        view_state = np.zeros_like(self.mean, dtype=np.uint8)
        view_state[self.blocked_count > 0] = VIEW_BLOCKED_ONLY
        view_state[self.visit_count > 0] = VIEW_CLEAR
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
            view_state=view_state,
            view_sectors=self.view_sectors.copy(),
        )
