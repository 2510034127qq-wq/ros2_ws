"""Information-gain target selection for thermal exploration."""

from __future__ import annotations

import math
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


@dataclass
class PlannerSector:
    yaw: float
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
    preferred_yaw: Optional[float] = None,
    directional_weight: float = 0.0,
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
    directional_gain = np.zeros_like(dist, dtype=np.float32)
    if preferred_yaw is not None and directional_weight > 0.0:
        target_yaw = np.arctan2(wy - robot_wy, wx - robot_wx)
        align = 0.5 + 0.5 * np.cos(target_yaw - float(preferred_yaw))
        directional_gain = align.astype(np.float32)

    score = (
        weights.information_gain * information_gain
        + weights.source_probability * source_probability
        + weights.coverage_gain * coverage_gain
        + float(directional_weight) * directional_gain
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


def select_coverage_ring_target(
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
    min_radius: float = 4.0,
    max_radius: float = 9.0,
    safe_dist: float = 2.3,
    weights: PlannerWeights = PlannerWeights(),
    preferred_yaw: Optional[float] = None,
    directional_weight: float = 0.0,
    footprint_radius: float = 3.0,
    num_angles: int = 24,
    num_rings: int = 3,
    recent_yaws: Sequence[float] = (),
    recent_yaw_penalty: float = 0.0,
    anchor_x: Optional[float] = None,
    anchor_y: Optional[float] = None,
    min_travel_d: Optional[float] = None,
    max_travel_d: Optional[float] = None,
    angle_span_rad: Optional[float] = None,
) -> Optional[PlannerTarget]:
    """Select a medium-range sweep target by expected FOV information gain.

    Cell-wise frontiers can over-focus on a single high-scoring map cell.  This
    routine scores where the robot's next thermal view footprint would land, so
    a target is valuable only if it exposes an under-observed local area while
    staying clear of already confirmed sources.
    """
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return None
    if max_radius < min_radius or num_angles <= 0 or num_rings <= 0:
        return None

    variance = _reshape(temperature_variance, height, width)
    conf = _reshape(confidence, height, width)
    visits = _reshape(visit_count, height, width).astype(np.float32)
    age = _reshape(last_seen_age_s, height, width).astype(np.float32)

    yy, xx = np.mgrid[0:height, 0:width]
    wx = origin_x + (xx.astype(np.float32) + 0.5) * resolution
    wy = origin_y + (yy.astype(np.float32) + 0.5) * resolution

    var_norm = _norm_clip(variance)
    unseen = 1.0 - np.clip(conf, 0.0, 1.0)
    age_norm = np.where(age >= 0.0, np.clip(age / 60.0, 0.0, 1.0), 1.0)
    information_gain = 0.45 * unseen + 0.35 * var_norm + 0.20 * age_norm
    coverage_gain = 1.0 / (1.0 + visits)

    source_probability = np.zeros((height, width), dtype=np.float32)
    for src in source_estimates:
        if src.status in ("suppressed", "stale"):
            continue
        spread = 1.8 if src.status == "candidate" else 1.2
        d = np.sqrt((wx - src.x) ** 2 + (wy - src.y) ** 2)
        amp = max(0.0, min(1.0, src.probability)) * max(0.25, min(1.0, src.confidence or src.probability))
        source_probability = np.maximum(source_probability, amp * np.exp(-0.5 * (d / spread) ** 2))

    duplicate_cell = np.zeros((height, width), dtype=np.float32)
    for sx, sy in known_sources:
        d = np.sqrt((wx - sx) ** 2 + (wy - sy) ** 2)
        duplicate_cell = np.maximum(duplicate_cell, np.exp(-0.5 * (d / max(safe_dist, 0.25)) ** 2))

    cell_value = (
        weights.information_gain * information_gain
        + weights.source_probability * source_probability
        + weights.coverage_gain * coverage_gain
        - weights.duplicate_penalty * duplicate_cell
    )

    ax = robot_wx if anchor_x is None else float(anchor_x)
    ay = robot_wy if anchor_y is None else float(anchor_y)
    base_yaw = 0.0 if preferred_yaw is None else float(preferred_yaw)
    footprint_radius = max(float(footprint_radius), resolution)
    map_min_x = origin_x
    map_min_y = origin_y
    map_max_x = origin_x + width * resolution
    map_max_y = origin_y + height * resolution
    min_travel = 0.0 if min_travel_d is None else float(min_travel_d)
    max_travel = float("inf") if max_travel_d is None else float(max_travel_d)

    best: Optional[PlannerTarget] = None
    radii = np.linspace(float(min_radius), float(max_radius), max(1, int(num_rings)))
    if angle_span_rad is None or float(angle_span_rad) >= math.pi * 2.0:
        yaw_offsets = [idx * 2.0 * math.pi / float(num_angles) for idx in range(int(num_angles))]
    elif int(num_angles) <= 1:
        yaw_offsets = [0.0]
    else:
        span = max(0.0, float(angle_span_rad))
        yaw_offsets = np.linspace(-span, span, int(num_angles))
    for radius in radii:
        for offset in yaw_offsets:
            yaw = _wrap_angle(base_yaw + float(offset))
            tx = ax + float(radius) * math.cos(yaw)
            ty = ay + float(radius) * math.sin(yaw)
            if tx < map_min_x or tx > map_max_x or ty < map_min_y or ty > map_max_y:
                continue
            travel_d = math.hypot(tx - robot_wx, ty - robot_wy)
            if travel_d < min_travel or travel_d > max_travel:
                continue

            duplicate_target = 0.0
            too_close = False
            for sx, sy in known_sources:
                d_src = math.hypot(tx - sx, ty - sy)
                if d_src < safe_dist:
                    too_close = True
                    break
                duplicate_target = max(
                    duplicate_target,
                    math.exp(-0.5 * (d_src / max(safe_dist, 0.25)) ** 2),
                )
            if too_close:
                continue

            footprint = ((wx - tx) ** 2 + (wy - ty) ** 2) <= footprint_radius ** 2
            for sx, sy in known_sources:
                footprint &= ((wx - sx) ** 2 + (wy - sy) ** 2) >= safe_dist ** 2
            if not np.any(footprint):
                continue

            values = cell_value[footprint]
            cutoff = float(np.percentile(values, 75.0))
            top = values[values >= cutoff]
            top_mean = float(np.mean(top)) if top.size else float(np.mean(values))
            mean_value = float(np.mean(values))
            uncertain_area = float(np.mean(
                (unseen[footprint] > 0.35)
                | (coverage_gain[footprint] > 0.5)
                | (age_norm[footprint] > 0.5)
            ))
            fov_gain = 0.62 * top_mean + 0.25 * mean_value + 0.13 * uncertain_area

            directional_gain = 0.0
            if preferred_yaw is not None and directional_weight > 0.0:
                directional_gain = 0.5 + 0.5 * math.cos(_wrap_angle(yaw - float(preferred_yaw)))

            recent_penalty = 0.0
            if recent_yaw_penalty > 0.0:
                for recent_yaw in recent_yaws:
                    align = 0.5 + 0.5 * math.cos(_wrap_angle(yaw - float(recent_yaw)))
                    recent_penalty = max(recent_penalty, align * align)

            edge_m = min(tx - map_min_x, map_max_x - tx, ty - map_min_y, map_max_y - ty)
            edge_penalty = max(0.0, 1.0 - edge_m / max(footprint_radius, resolution))
            travel_cost = min(1.0, travel_d / max(max_travel if math.isfinite(max_travel) else max_radius, 1e-3))
            score = (
                fov_gain
                + float(directional_weight) * directional_gain
                - 0.25 * travel_cost
                - weights.duplicate_penalty * duplicate_target
                - weights.risk_penalty * edge_penalty
                - float(recent_yaw_penalty) * recent_penalty
            )

            source_peak = float(np.max(source_probability[footprint])) if np.any(footprint) else 0.0
            if source_peak > 0.15:
                reason = "source_verify_ring"
            elif anchor_x is not None or anchor_y is not None:
                reason = "annular_coverage"
            else:
                reason = "coverage_ring"
            if best is None or score > best.score:
                best = PlannerTarget(x=float(tx), y=float(ty), score=float(score), reason=reason)
    return best


def select_exploration_sector_yaw(
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
    known_sources: Sequence[Tuple[float, float]] = (),
    min_d: float = 3.0,
    max_d: float = 16.0,
    safe_dist: float = 2.3,
    phase_yaw: float = 0.0,
    num_sectors: int = 16,
) -> Optional[PlannerSector]:
    """Pick an exploration direction from map evidence, not scenario layout.

    The phase is only a deterministic tie-breaker for equally unknown maps.  The
    main score is uncertainty, low confidence, low visit count, age, travel cost,
    and distance from confirmed sources.
    """
    if width <= 0 or height <= 0 or num_sectors <= 0:
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
    travel_cost = np.clip(dist / max(max_d, 1e-3), 0.0, 1.0)
    cell_score = 0.58 * information_gain + 0.42 * coverage_gain - 0.35 * travel_cost - 0.85 * duplicate
    angle = np.arctan2(wy - robot_wy, wx - robot_wx)
    sector_width = math.pi / max(1, num_sectors)
    valid_count = float(np.count_nonzero(valid))

    best: Optional[PlannerSector] = None
    for idx in range(num_sectors):
        yaw = _wrap_angle(float(phase_yaw) + idx * 2.0 * math.pi / num_sectors)
        diff = np.abs(np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw)))
        mask = valid & (diff <= sector_width)
        if not np.any(mask):
            continue
        values = cell_score[mask]
        cutoff = float(np.percentile(values, 80.0))
        top = values[values >= cutoff]
        top_mean = float(np.mean(top)) if top.size else float(np.mean(values))
        sector_mean = float(np.mean(values))
        area_gain = float(np.count_nonzero(mask)) / max(1.0, valid_count)
        tie_break = 0.01 * math.cos(yaw - float(phase_yaw))
        score = 0.72 * top_mean + 0.18 * sector_mean + 0.10 * area_gain + tie_break
        if best is None or score > best.score:
            best = PlannerSector(yaw=yaw, score=score, reason="sector_information")
    return best


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


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def select_residual_target(
    robot_wx: float,
    robot_wy: float,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    residual: np.ndarray,
    view_state: np.ndarray,
    last_seen_age_s: np.ndarray,
    known_sources: Sequence[Tuple[float, float]] = (),
    min_d: float = 2.5,
    max_d: float = 16.0,
    safe_dist: float = 2.3,
    w_residual: float = 1.6,
    w_unseen: float = 1.0,
    w_blocked: float = 1.2,
    w_age: float = 0.25,
    w_travel: float = 0.45,
    w_duplicate: float = 1.2,
    top_k: int = 12,
    line_reachable_fn=None,
) -> Optional[PlannerTarget]:
    """Select a residual-exploration target with optional straight-line reachability."""
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return None
    resid = _reshape(residual, height, width)
    vs = np.asarray(view_state).reshape((height, width))
    age = _reshape(last_seen_age_s, height, width)

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

    resid_positive = resid[resid > 0.0]
    if resid_positive.size:
        resid_scale = float(np.percentile(resid_positive, 95.0))
        resid_norm = np.clip(resid / max(resid_scale, 1e-6), 0.0, 1.0).astype(np.float32)
    else:
        resid_norm = np.zeros_like(resid, dtype=np.float32)
    unseen = (vs == 0).astype(np.float32)
    blocked = (vs == 1).astype(np.float32)
    age_norm = np.where((vs == 2) & (age >= 0.0),
                        np.clip(age / 60.0, 0.0, 1.0), 0.0).astype(np.float32)
    travel = np.clip(dist / max(max_d, 1e-3), 0.0, 1.0)

    term_resid = w_residual * resid_norm
    term_unseen = w_unseen * unseen
    term_blocked = w_blocked * blocked
    score = (term_resid + term_unseen + term_blocked + w_age * age_norm
             - w_travel * travel - w_duplicate * duplicate)
    score[~valid] = -np.inf
    if not np.isfinite(score).any():
        return None

    flat_score = score.reshape(-1)
    flat_order = np.argsort(flat_score)[::-1][:max(1, int(top_k))]

    def _mk(idx: int, reachable: bool) -> PlannerTarget:
        iy, ix = np.unravel_index(int(idx), score.shape)
        terms = {
            "residual_mass": float(term_resid[iy, ix]),
            "unseen": float(term_unseen[iy, ix]),
            "blocked_view": float(term_blocked[iy, ix]),
        }
        reason = max(terms, key=terms.get)
        target = PlannerTarget(
            x=float(origin_x + (ix + 0.5) * resolution),
            y=float(origin_y + (iy + 0.5) * resolution),
            score=float(score[iy, ix]),
            reason=reason,
        )
        target.metadata_reachable = reachable
        return target

    if line_reachable_fn is None:
        return _mk(int(flat_order[0]), True)
    for idx in flat_order:
        if not np.isfinite(flat_score[int(idx)]):
            break
        iy, ix = np.unravel_index(int(idx), score.shape)
        tx = float(origin_x + (ix + 0.5) * resolution)
        ty = float(origin_y + (iy + 0.5) * resolution)
        if line_reachable_fn(robot_wx, robot_wy, tx, ty):
            return _mk(int(idx), True)
    return _mk(int(flat_order[0]), False)
