"""Information-gain target selection for thermal exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PlannerWeights:
    information_gain: float = 1.0
    source_probability: float = 1.4
    coverage_gain: float = 0.7
    travel_cost: float = 0.45
    duplicate_penalty: float = 1.2
    risk_penalty: float = 0.2


@dataclass
class PlannerSource:
    x: float
    y: float
    probability: float
    status: str = "candidate"
    confidence: float = 0.0


@dataclass
class PlannerTarget:
    x: float
    y: float
    score: float
    reason: str


def select_information_gain_target(
    robot_wx: float,
    robot_wy: float,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    temperature_variance: np.ndarray,
    confidence: np.ndarray,
    visit_count: np.ndarray,
    last_seen_age_s: np.ndarray,
    source_estimates: Sequence[PlannerSource] = (),
    known_sources: Sequence[Tuple[float, float]] = (),
    min_d: float = 3.0,
    max_d: float = 16.0,
    safe_dist: float = 2.3,
    weights: PlannerWeights = PlannerWeights(),
) -> Optional[PlannerTarget]:
    if width <= 0 or height <= 0:
        return None
    variance = _reshape(temperature_variance, height, width)
    conf = _reshape(confidence, height, width)
    visits = _reshape(visit_count, height, width).astype(np.float32)
    age = _reshape(last_seen_age_s, height, width).astype(np.float32)

    yy, xx = np.mgrid[0:height, 0:width]
    wx = origin_x + (xx.astype(np.float32) + 0.5) * resolution
    wy = origin_y + (yy.astype(np.float32) + 0.5) * resolution
    dist = np.sqrt((wx - robot_wx) ** 2 + (wy - robot_wy) ** 2)
    valid = (dist >= min_d) & (dist <= max_d)

    duplicate = np.zeros_like(dist, dtype=np.float32)
    for sx, sy in known_sources:
        d = np.sqrt((wx - sx) ** 2 + (wy - sy) ** 2)
        valid &= d >= safe_dist
        duplicate = np.maximum(duplicate, np.exp(-0.5 * (d / max(safe_dist, 0.25)) ** 2))

    if not np.any(valid):
        return None

    var_norm = _norm_clip(variance)
    unseen = 1.0 - np.clip(conf, 0.0, 1.0)
    age_norm = np.where(age >= 0.0, np.clip(age / 60.0, 0.0, 1.0), 1.0)
    information_gain = 0.45 * unseen + 0.35 * var_norm + 0.20 * age_norm
    coverage_gain = 1.0 / (1.0 + visits)

    source_probability = np.zeros_like(dist, dtype=np.float32)
    for src in source_estimates:
        if src.status in ("suppressed", "stale"):
            continue
        spread = 1.8 if src.status == "candidate" else 1.2
        d = np.sqrt((wx - src.x) ** 2 + (wy - src.y) ** 2)
        amp = max(0.0, min(1.0, src.probability)) * max(0.25, min(1.0, src.confidence or src.probability))
        source_probability = np.maximum(source_probability, amp * np.exp(-0.5 * (d / spread) ** 2))

    travel_cost = np.clip(dist / max(max_d, 1e-3), 0.0, 1.0)
    risk_penalty = np.zeros_like(dist, dtype=np.float32)
    map_margin = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    risk_penalty += np.clip(1.0 - map_margin / 4.0, 0.0, 1.0) * 0.5

    score = (
        weights.information_gain * information_gain
        + weights.source_probability * source_probability
        + weights.coverage_gain * coverage_gain
        - weights.travel_cost * travel_cost
        - weights.duplicate_penalty * duplicate
        - weights.risk_penalty * risk_penalty
    )
    score[~valid] = -np.inf
    if not np.isfinite(score).any():
        return None
    iy, ix = np.unravel_index(int(np.nanargmax(score)), score.shape)
    reason = "source_verify" if source_probability[iy, ix] > 0.15 else (
        "coverage" if coverage_gain[iy, ix] > information_gain[iy, ix] else "information_gain"
    )
    return PlannerTarget(
        x=float(origin_x + (ix + 0.5) * resolution),
        y=float(origin_y + (iy + 0.5) * resolution),
        score=float(score[iy, ix]),
        reason=reason,
    )


def _reshape(values: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape == (height, width):
        return arr.astype(np.float32, copy=False)
    return arr.reshape((height, width)).astype(np.float32, copy=False)


def _norm_clip(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    finite = np.where(np.isfinite(finite), finite, 0.0)
    p95 = float(np.percentile(finite, 95)) if finite.size else 1.0
    if p95 <= 1e-6:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip(finite / p95, 0.0, 1.0).astype(np.float32)
