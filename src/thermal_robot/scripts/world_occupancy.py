#!/usr/bin/env python3
"""Ground-truth occupancy grids from Gazebo SDF worlds, with line-of-sight checks."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

GRID_CENTER_X = -6.0
GRID_CENTER_Y = 0.0
GRID_SIZE_M = 50.0
GRID_RESOLUTION = 0.1
OBSTACLE_Z_MIN = 0.15
OBSTACLE_Z_MAX = 1.5
LOS_STEP_M = 0.05


@dataclass(frozen=True)
class WorldBox:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    size_z: float


@dataclass
class TruthOccupancyGrid:
    resolution: float
    origin_x: float
    origin_y: float
    data: np.ndarray


def _parse_pose(text: str) -> Tuple[float, float, float, float]:
    parts = [float(v) for v in (text or '0 0 0 0 0 0').split()]
    while len(parts) < 6:
        parts.append(0.0)
    return parts[0], parts[1], parts[2], parts[5]


def parse_world_boxes(sdf_text: str) -> List[WorldBox]:
    root = ET.fromstring(sdf_text)
    boxes: List[WorldBox] = []
    for model in root.iter('model'):
        mx, my, mz, myaw = _parse_pose(model.findtext('pose', default='0 0 0 0 0 0'))
        for link in model.findall('link'):
            lx, ly, lz, lyaw = _parse_pose(link.findtext('pose', default='0 0 0 0 0 0'))
            for coll in link.findall('collision'):
                cx, cy, cz, cyaw = _parse_pose(coll.findtext('pose', default='0 0 0 0 0 0'))
                box = coll.find('geometry/box/size')
                if box is None or box.text is None:
                    continue
                sx, sy, sz = (float(v) for v in box.text.split()[:3])
                ox, oy = lx + cx, ly + cy
                wx = mx + math.cos(myaw) * ox - math.sin(myaw) * oy
                wy = my + math.sin(myaw) * ox + math.cos(myaw) * oy
                wz = mz + lz + cz
                yaw = myaw + lyaw + cyaw
                z_lo, z_hi = wz - sz / 2.0, wz + sz / 2.0
                if z_hi < OBSTACLE_Z_MIN or z_lo > OBSTACLE_Z_MAX:
                    continue
                boxes.append(WorldBox(model.get('name', 'unnamed'),
                                      wx, wy, wz, yaw, sx, sy, sz))
    return boxes


def rasterize(boxes: List[WorldBox],
              resolution: float = GRID_RESOLUTION,
              center_x: float = GRID_CENTER_X,
              center_y: float = GRID_CENTER_Y,
              size_m: float = GRID_SIZE_M) -> TruthOccupancyGrid:
    n = int(round(size_m / resolution))
    origin_x = center_x - size_m / 2.0
    origin_y = center_y - size_m / 2.0
    data = np.zeros((n, n), dtype=np.uint8)
    ys, xs = np.mgrid[0:n, 0:n]
    cell_wx = origin_x + (xs + 0.5) * resolution
    cell_wy = origin_y + (ys + 0.5) * resolution
    for box in boxes:
        dx = cell_wx - box.x
        dy = cell_wy - box.y
        cos_y, sin_y = math.cos(-box.yaw), math.sin(-box.yaw)
        u = cos_y * dx - sin_y * dy
        v = sin_y * dx + cos_y * dy
        inside = (np.abs(u) <= box.size_x / 2.0) & (np.abs(v) <= box.size_y / 2.0)
        data[inside] = 1
    return TruthOccupancyGrid(resolution=resolution, origin_x=origin_x,
                              origin_y=origin_y, data=data)


def is_occupied(grid: TruthOccupancyGrid, wx: float, wy: float) -> bool:
    ix = int(math.floor((wx - grid.origin_x) / grid.resolution))
    iy = int(math.floor((wy - grid.origin_y) / grid.resolution))
    h, w = grid.data.shape
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return False
    return bool(grid.data[iy, ix])


def line_of_sight(grid: TruthOccupancyGrid,
                  x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the segment crosses no occupied cell."""
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / LOS_STEP_M))
    for i in range(steps + 1):
        u = i / steps
        if is_occupied(grid, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u):
            return False
    return True


def save_grid(grid: TruthOccupancyGrid, path: Path) -> None:
    np.savez_compressed(path, data=grid.data,
                        resolution=grid.resolution,
                        origin_x=grid.origin_x, origin_y=grid.origin_y)


def load_grid(path: Path) -> TruthOccupancyGrid:
    with np.load(path) as f:
        return TruthOccupancyGrid(
            resolution=float(f['resolution']),
            origin_x=float(f['origin_x']),
            origin_y=float(f['origin_y']),
            data=f['data'].astype(np.uint8),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds-dir', default='', help='directory of .world files')
    parser.add_argument('--world', default='', help='single .world file')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    worlds: List[Path] = []
    if args.world:
        worlds.append(Path(args.world).expanduser())
    if args.worlds_dir:
        worlds.extend(sorted(Path(args.worlds_dir).expanduser().glob('*.world')))
    if not worlds:
        parser.error('need --world or --worlds-dir')

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for world in worlds:
        boxes = parse_world_boxes(world.read_text())
        grid = rasterize(boxes)
        out = out_dir / f'{world.stem}.npz'
        save_grid(grid, out)
        print(f'[occupancy] {world.name}: {len(boxes)} boxes, '
              f'{int(grid.data.sum())} occupied cells -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
