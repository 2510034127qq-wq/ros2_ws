"""Ray-cast visibility against occupancy grids, independent of ROS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_OCCUPIED_THRESHOLD = 65
DEFAULT_RAY_STEP_M = 0.1


def valid_occupancy_grid_shape(data, width: int, height: int) -> bool:
    """Validate an OccupancyGrid payload before reshaping or caching it."""
    width = int(width)
    height = int(height)
    return width > 0 and height > 0 and len(data) == width * height


def occupancy_grid_has_known_cells(data, width: int, height: int) -> bool:
    """Return true when a well-shaped map contains free or occupied evidence."""
    if not valid_occupancy_grid_shape(data, width, height):
        return False
    return bool(np.any(np.asarray(data, dtype=np.int16) >= 0))


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


def known_free_at(view: OccupancyView, wx, wy) -> np.ndarray:
    """Return true only for in-bounds cells explicitly observed as free."""
    wx = np.asarray(wx, dtype=np.float32)
    wy = np.asarray(wy, dtype=np.float32)
    ix = np.floor((wx - view.origin_x) / view.resolution).astype(np.int32)
    iy = np.floor((wy - view.origin_y) / view.resolution).astype(np.int32)
    h, w = view.data.shape
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.zeros(wx.shape, dtype=bool)
    if inside.any():
        vals = view.data[iy[inside], ix[inside]]
        out[inside] = (vals >= 0) & (vals < view.occupied_threshold)
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
    effective_step = max(
        1e-3,
        min(float(step_m), max(float(view.resolution), 1e-3) * 0.5),
    )
    if max_dist <= effective_step:
        return np.ones(txs.size, dtype=bool)
    n_steps = int(math.ceil(max_dist / effective_step))
    ts = np.arange(1, n_steps + 1, dtype=np.float32)[:, None] / float(n_steps)
    px = float(ox) + ts * dx[None, :]
    py = float(oy) + ts * dy[None, :]
    travelled = ts * dist[None, :]
    check = travelled < (dist[None, :] - 0.5 * effective_step)
    blocked = occupied_at(view, px, py) & check
    return ~blocked.any(axis=0)


def _segment_cells(view: OccupancyView, x0: float, y0: float,
                   x1: float, y1: float):
    """Return a conservative supercover of grid cells touched by a segment."""
    resolution = float(view.resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        return []
    gx0 = (float(x0) - view.origin_x) / resolution
    gy0 = (float(y0) - view.origin_y) / resolution
    gx1 = (float(x1) - view.origin_x) / resolution
    gy1 = (float(y1) - view.origin_y) / resolution
    if not all(math.isfinite(value) for value in (gx0, gy0, gx1, gy1)):
        return []

    ix, iy = int(math.floor(gx0)), int(math.floor(gy0))
    end_ix, end_iy = int(math.floor(gx1)), int(math.floor(gy1))
    dx, dy = gx1 - gx0, gy1 - gy0
    step_x = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    step_y = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
    if step_x:
        boundary_x = (ix + 1.0) if step_x > 0 else float(ix)
        t_max_x = (boundary_x - gx0) / dx
        t_delta_x = 1.0 / abs(dx)
    else:
        t_max_x = t_delta_x = float("inf")
    if step_y:
        boundary_y = (iy + 1.0) if step_y > 0 else float(iy)
        t_max_y = (boundary_y - gy0) / dy
        t_delta_y = 1.0 / abs(dy)
    else:
        t_max_y = t_delta_y = float("inf")

    cells = []
    seen = set()

    def add(cx, cy):
        key = (int(cy), int(cx))
        if key not in seen:
            seen.add(key)
            cells.append(key)

    add(ix, iy)
    max_steps = abs(end_ix - ix) + abs(end_iy - iy) + 4
    for _ in range(max_steps):
        if ix == end_ix and iy == end_iy:
            break
        tolerance = 1e-12 * max(1.0, abs(t_max_x), abs(t_max_y))
        if abs(t_max_x - t_max_y) <= tolerance:
            add(ix + step_x, iy)
            add(ix, iy + step_y)
            ix += step_x
            iy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
            add(ix, iy)
        elif t_max_x < t_max_y:
            ix += step_x
            t_max_x += t_delta_x
            add(ix, iy)
        else:
            iy += step_y
            t_max_y += t_delta_y
            add(ix, iy)
    return cells


def _segment_values(view: OccupancyView, x0: float, y0: float,
                    x1: float, y1: float):
    cells = _segment_cells(view, x0, y0, x1, y1)
    if not cells:
        return None
    height, width = view.data.shape
    if any(row < 0 or row >= height or col < 0 or col >= width
           for row, col in cells):
        return None
    return np.asarray([view.data[row, col] for row, col in cells], dtype=np.int16)


def line_reachable(view, x0: float, y0: float, x1: float, y1: float,
                   step_m: float = 0.2) -> bool:
    if view is None:
        return True
    values = _segment_values(view, x0, y0, x1, y1)
    return values is not None and bool(np.all(values < view.occupied_threshold))


def line_reachable_plannable(view, x0: float, y0: float, x1: float, y1: float,
                             step_m: float = 0.2) -> bool:
    """Require a known-free endpoint and no known occupied cell on the path.

    This is suitable for a Nav2-preferred frontier target: unknown intermediate
    space may become mapped during approach, but an occupied, unknown, or
    out-of-bounds endpoint is never accepted as the goal itself.
    """
    if view is None:
        return False
    endpoint_free = known_free_at(
        view,
        np.array([x1], dtype=np.float32),
        np.array([y1], dtype=np.float32),
    )
    if not bool(endpoint_free[0]):
        return False
    values = _segment_values(view, x0, y0, x1, y1)
    return values is not None and bool(np.all(values < view.occupied_threshold))


def line_reachable_known_free(view, x0: float, y0: float, x1: float, y1: float,
                              step_m: float = 0.2) -> bool:
    """Require the complete motion segment, including endpoint, to be known free.

    Visibility ray casting intentionally treats unknown space as transparent.
    Motion fallback cannot use that convention: unknown or out-of-bounds cells
    must not be interpreted as a collision-free direct path.
    """
    if view is None:
        return False
    values = _segment_values(view, x0, y0, x1, y1)
    return values is not None and bool(np.all(
        (values >= 0) & (values < view.occupied_threshold)))
