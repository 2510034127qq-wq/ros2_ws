"""Ray-cast visibility against occupancy grids, independent of ROS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_OCCUPIED_THRESHOLD = 65
DEFAULT_RAY_STEP_M = 0.1


@dataclass
class OccupancyView:
    origin_x: float
    origin_y: float
    resolution: float
    data: np.ndarray
    occupied_threshold: int = DEFAULT_OCCUPIED_THRESHOLD


def from_flat(data, width, height, origin_x, origin_y, resolution,
              occupied_threshold=DEFAULT_OCCUPIED_THRESHOLD) -> OccupancyView:
    arr = np.asarray(data, dtype=np.int16).reshape((int(height), int(width)))
    return OccupancyView(float(origin_x), float(origin_y), float(resolution),
                         arr, int(occupied_threshold))


def occupied_at(view: OccupancyView, wx, wy) -> np.ndarray:
    """Vectorized occupancy query; unknown and out-of-bounds cells are free."""
    wx = np.asarray(wx, dtype=np.float32)
    wy = np.asarray(wy, dtype=np.float32)
    ix = np.floor((wx - view.origin_x) / view.resolution).astype(np.int32)
    iy = np.floor((wy - view.origin_y) / view.resolution).astype(np.int32)
    h, w = view.data.shape
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.zeros(wx.shape, dtype=bool)
    if inside.any():
        vals = view.data[iy[inside], ix[inside]]
        out[inside] = vals >= view.occupied_threshold
    return out


def visible_mask(view: OccupancyView, ox: float, oy: float,
                 txs, tys, step_m: float = DEFAULT_RAY_STEP_M) -> np.ndarray:
    """Return True for targets whose segment from origin crosses no occupied cell."""
    txs = np.asarray(txs, dtype=np.float32).reshape(-1)
    tys = np.asarray(tys, dtype=np.float32).reshape(-1)
    if txs.size == 0:
        return np.zeros(0, dtype=bool)
    dx = txs - float(ox)
    dy = tys - float(oy)
    dist = np.hypot(dx, dy)
    max_dist = float(dist.max())
    if max_dist <= step_m:
        return np.ones(txs.size, dtype=bool)
    n_steps = int(math.ceil(max_dist / step_m))
    ts = np.arange(1, n_steps + 1, dtype=np.float32)[:, None] / float(n_steps)
    px = float(ox) + ts * dx[None, :]
    py = float(oy) + ts * dy[None, :]
    travelled = ts * dist[None, :]
    check = travelled < (dist[None, :] - 0.5 * step_m)
    blocked = occupied_at(view, px, py) & check
    return ~blocked.any(axis=0)


def line_reachable(view, x0: float, y0: float, x1: float, y1: float,
                   step_m: float = 0.2) -> bool:
    if view is None:
        return True
    return bool(visible_mask(view, x0, y0,
                             np.array([x1], dtype=np.float32),
                             np.array([y1], dtype=np.float32),
                             step_m=step_m)[0])
